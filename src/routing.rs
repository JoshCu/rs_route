use crate::config::ChannelParams;
use crate::io::csv::load_external_flows;
use crate::io::netcdf::write_batch;
use crate::io::results::SimulationResults;
use crate::kernel::muskingum::rs_route::mc_kernel_simd::{self, LANES, LaneParams};
use crate::kernel::muskingum::{
    MuskingumCungeInput, MuskingumCungeKernel, MuskingumCungeResult, SecantBracket,
};
use crate::network::NetworkTopology;
use anyhow::{Context, Result};
use indicatif::ProgressBar;
use netcdf::FileMut;
use rustc_hash::FxHashMap;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

// Message types
enum WriterMessage {
    WriteResults(Arc<SimulationResults>),
    Shutdown,
}

enum WorkerMessage {
    ProcessNode(u32),
    Shutdown,
}

enum SchedulerMessage {
    NodeCompleted(u32),
    Shutdown,
}

/// A node's resolved inputs: the upstream hydrograph and the lateral-flow
/// series, both indexed by internal timestep.
struct NodeWork {
    inflow: Vec<f32>,
    external: Vec<f32>,
    /// Internal timesteps per external (forcing) timestep.
    upsampling: usize,
}

/// Gather a node's inputs.
///
/// Returns `None` when the node has neither upstream inflow nor lateral flow,
/// which routes to all zeros.
fn prepare_node(
    node_id: &u32,
    topology: &NetworkTopology,
    max_timesteps: usize,
) -> Result<Option<NodeWork>> {
    let node = topology
        .nodes
        .get(node_id)
        .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

    let area = node
        .area_sqkm
        .ok_or_else(|| anyhow::anyhow!("Node {} has no area defined", node_id))?;

    let mut external_flows =
        load_external_flows(node.qlat_file.clone(), &node.id, Some(&"Q_OUT"), area)?;

    // The scheduler only releases a node once every upstream has completed, so
    // nothing can still be writing here. Take the buffer instead of holding the
    // lock for the whole routing pass; the caller used to clear it afterwards.
    let mut inflow = {
        let mut guard = node
            .inflow_storage
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock inflow storage: {}", e))?;
        std::mem::take(&mut *guard)
    };

    if inflow.is_empty() && external_flows.is_empty() {
        return Ok(None);
    }

    // if headwater then upstream inflow is 0.0
    if inflow.is_empty() {
        inflow.resize(max_timesteps, 0.0);
    }

    if external_flows.is_empty() {
        external_flows.resize(max_timesteps, 0.0);
    } else if external_flows.len() == 1 {
        // Only a single external flow value breaks the upsampling logic,
        // so we throw an error if the file only contains one value (which is likely a mistake)
        return Err(anyhow::anyhow!(
            "External flow file for node {} only contains one value, which is not sufficient for routing. Please check the file: {:?}",
            node_id,
            node.qlat_file
        )).with_context(|| format!("Failed to load external flows for node {}: {:?}", node_id, node.qlat_file));
    }

    // -1 because the input files have one additional timestep
    let upsampling = (max_timesteps / (external_flows.len() - 1)).max(1);

    Ok(Some(NodeWork {
        inflow: Vec::from(inflow),
        external: Vec::from(external_flows),
        upsampling,
    }))
}

fn zero_results(feature_id: u32, max_timesteps: usize) -> SimulationResults {
    let mut results = SimulationResults::new(feature_id);
    results.flow_data = vec![0.0; max_timesteps];
    results.velocity_data = vec![0.0; max_timesteps];
    results.depth_data = vec![0.0; max_timesteps];
    results
}

/// Route one node, one timestep at a time, through the selected scalar kernel.
fn route_node_scalar(
    kernel: MuskingumCungeKernel,
    feature_id: u32,
    work: &NodeWork,
    channel_params: &ChannelParams,
    max_timesteps: usize,
    dt: f32,
    bracket: SecantBracket,
) -> SimulationResults {
    let mut results = SimulationResults::new(feature_id);
    results.flow_data.reserve(max_timesteps);
    results.velocity_data.reserve(max_timesteps);
    results.depth_data.reserve(max_timesteps);

    let s0 = if channel_params.s0 == 0.0 {
        0.00001
    } else {
        channel_params.s0
    };

    let mut qup = 0.0;
    let mut qdp = 0.0;
    let mut depth_p = 0.0;

    for timestep in 0..max_timesteps {
        let upstream_flow = work.inflow.get(timestep).copied().unwrap_or(0.0);
        let external_flow = work
            .external
            .get(timestep / work.upsampling)
            .copied()
            .unwrap_or(0.0);

        let result: MuskingumCungeResult = kernel.exec_with_bracket(
            &MuskingumCungeInput {
                dt,
                qup,
                quc: upstream_flow,
                qdp,
                ql: external_flow,
                dx: channel_params.dx,
                bw: channel_params.bw,
                tw: channel_params.tw,
                tw_cc: channel_params.twcc,
                n: channel_params.n,
                n_cc: channel_params.ncc,
                cs: channel_params.cs,
                s0,
                velp: 0.0, // unused
                depthp: depth_p,
            },
            false,
            bracket,
        );

        results.flow_data.push(result.qdc);
        results.velocity_data.push(result.velc);
        results.depth_data.push(result.depthc);

        qup = upstream_flow;
        qdp = result.qdc;
        depth_p = result.depthc;
    }

    results
}

/// Route up to `LANES` independent nodes together, one timestep at a time.
///
/// The reaches in a batch have no dependency on each other -- the scheduler
/// only ever releases nodes whose upstreams are all complete -- so they can be
/// stepped in lockstep through the SIMD kernel. Short batches leave the spare
/// lanes at zero flow, which the kernel retires immediately.
fn route_nodes_simd(
    batch: &[(u32, NodeWork, ChannelParams)],
    max_timesteps: usize,
    dt: f32,
    bracket: SecantBracket,
) -> Vec<SimulationResults> {
    let params: Vec<ChannelParams> = batch.iter().map(|(_, _, p)| p.clone()).collect();
    let lane_params = LaneParams::build(dt, &params);

    let mut results: Vec<SimulationResults> = batch
        .iter()
        .map(|(id, _, _)| {
            let mut r = SimulationResults::new(*id);
            r.flow_data.reserve(max_timesteps);
            r.velocity_data.reserve(max_timesteps);
            r.depth_data.reserve(max_timesteps);
            r
        })
        .collect();

    let mut qup = [0.0f32; LANES];
    let mut qdp = [0.0f32; LANES];
    let mut depth_p = [0.0f32; LANES];

    for timestep in 0..max_timesteps {
        let mut quc = [0.0f32; LANES];
        let mut ql = [0.0f32; LANES];
        for (lane, (_, work, _)) in batch.iter().enumerate() {
            quc[lane] = work.inflow.get(timestep).copied().unwrap_or(0.0);
            ql[lane] = work
                .external
                .get(timestep / work.upsampling)
                .copied()
                .unwrap_or(0.0);
        }

        let (out, _iters) =
            mc_kernel_simd::step(&lane_params, &qup, &quc, &qdp, &ql, &depth_p, bracket);

        for lane in 0..batch.len() {
            results[lane].flow_data.push(out.qdc[lane]);
            results[lane].velocity_data.push(out.velc[lane]);
            results[lane].depth_data.push(out.depthc[lane]);
            qup[lane] = quc[lane];
            qdp[lane] = out.qdc[lane];
            depth_p[lane] = out.depthc[lane];
        }
    }

    results
}

fn writer_thread(
    receiver: Receiver<WriterMessage>,
    output_file: Arc<Mutex<FileMut>>,
    batch_size: usize, // e.g., 100 nodes
) -> Result<()> {
    let mut batch = Vec::new();
    let mut batch_num = 0;

    loop {
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(WriterMessage::WriteResults(results)) => {
                batch.push(results);

                // Write when batch is full
                if batch.len() >= batch_size {
                    write_batch(&output_file, &batch, batch_num)?;
                    batch.clear();
                    batch_num += 1;
                }
            }
            Ok(WriterMessage::Shutdown) => {
                // Write remaining batch
                if !batch.is_empty() {
                    write_batch(&output_file, &batch, batch_num)?;
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Write partial batch on timeout to avoid holding data too long
                if !batch.is_empty() {
                    write_batch(&output_file, &batch, batch_num)?;
                    batch.clear();
                    batch_num += 1;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All senders dropped — normal shutdown
                if !batch.is_empty() {
                    write_batch(&output_file, &batch, batch_num)?;
                }
                break;
            }
        }
    }
    Ok(())
}

// Scheduler thread that tracks dependencies and sends ready work
fn scheduler_thread(
    topology: Arc<NetworkTopology>,
    scheduler_rx: Receiver<SchedulerMessage>,
    worker_tx: Vec<Sender<WorkerMessage>>,
) -> Result<()> {
    // Track which nodes are ready to process
    let mut ready_nodes = VecDeque::new();
    let mut pending_upstreams_count: FxHashMap<u32, usize> = FxHashMap::default();

    // Initialize with leaf nodes (no upstream dependencies)
    for node_id in topology.nodes.keys() {
        let upstream_count = topology.upstream_counts.get(node_id).copied().unwrap_or(0);
        if upstream_count == 0 {
            ready_nodes.push_back(*node_id);
        } else {
            // Count how many upstream nodes need to complete
            pending_upstreams_count.insert(*node_id, upstream_count);
        }
    }

    let num_workers = worker_tx.len();
    let mut next_worker = 0;
    let mut pending_runs = 0;

    loop {
        // Send ready work to workers
        while let Some(node_id) = ready_nodes.pop_front() {
            // Round-robin distribution to workers
            if let Err(e) = worker_tx[next_worker].send(WorkerMessage::ProcessNode(node_id)) {
                eprintln!("Failed to send work to worker {}: {}", next_worker, e);
            }
            next_worker = (next_worker + 1) % num_workers;
            pending_runs += 1;
        }

        // Wait for completion messages
        match scheduler_rx.recv() {
            Ok(SchedulerMessage::NodeCompleted(node_id)) => {
                pending_runs -= 1;
                // Check if this enables any downstream nodes
                if let Some(node) = topology.nodes.get(&node_id) {
                    if let Some(count) = pending_upstreams_count.get_mut(&node.downstream_id) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            // Prioritize downstream nodes to free inflow buffers sooner
                            ready_nodes.push_front(node.downstream_id);
                            pending_upstreams_count.remove(&node.downstream_id);
                        }
                    }
                }
            }
            Ok(SchedulerMessage::Shutdown) => break,
            Err(e) => {
                eprintln!("Scheduler channel error: {}", e);
                break;
            }
        }
        if pending_runs == 0 && ready_nodes.is_empty() {
            break;
        }
    }

    // Send shutdown to all workers
    for tx in &worker_tx {
        let _ = tx.send(WorkerMessage::Shutdown);
    }

    Ok(())
}

// Downsample full-resolution results to output frequency
fn downsample_results(results: SimulationResults, downsampling: usize) -> SimulationResults {
    if downsampling <= 1 {
        return results;
    }
    let actual_timesteps = results.flow_data.len();
    let mut flow_data = Vec::with_capacity(actual_timesteps / downsampling);
    let mut velocity_data = Vec::with_capacity(actual_timesteps / downsampling);
    let mut depth_data = Vec::with_capacity(actual_timesteps / downsampling);
    for i in (downsampling - 1..actual_timesteps).step_by(downsampling) {
        flow_data.push(results.flow_data[i]);
        velocity_data.push(results.velocity_data[i]);
        depth_data.push(results.depth_data[i]);
    }
    SimulationResults {
        feature_id: results.feature_id,
        flow_data,
        velocity_data,
        depth_data,
    }
}

// Worker thread - now just receives work and processes it
/// Hand a finished node's results downstream and on to the writer.
fn finish_node(
    node_id: u32,
    results: SimulationResults,
    topology: &NetworkTopology,
    downsampling: usize,
    writer_tx: &Sender<WriterMessage>,
) -> Result<()> {
    // Pass full-resolution flow to downstream node
    if let Some(node) = topology.nodes.get(&node_id) {
        if let Some(downstream_node) = topology.nodes.get(&node.downstream_id) {
            let mut buffer = downstream_node
                .inflow_storage
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock downstream buffer: {}", e))?;
            if buffer.is_empty() {
                buffer.resize(results.flow_data.len(), 0.0);
            }
            for (i, &flow) in results.flow_data.iter().enumerate() {
                if i < buffer.len() {
                    buffer[i] += flow;
                }
            }
        }
    }

    // Downsample then send to writer
    let downsampled = downsample_results(results, downsampling);
    if let Err(e) = writer_tx.send(WriterMessage::WriteResults(Arc::new(downsampled))) {
        eprintln!("Failed to send results to writer: {}", e);
    }
    Ok(())
}

fn report_node_error(node_id: u32, e: &anyhow::Error) {
    let mut error_message = format!("Error processing node {}: {}", node_id, e);
    // if error context, elaborate on it
    if let Some(context) = e.chain().nth(1) {
        error_message.push_str(&format!("\nContext: {}", context));
    }
    eprintln!("{}", error_message);
}

// Worker thread - receives ready nodes and routes them.
//
// The SIMD kernel routes `LANES` reaches at once, so under that kernel the
// worker drains whatever else is already queued (up to a full batch) before
// starting. Nodes arrive here only once all their upstreams are done, so
// everything in a batch is independent by construction. When the frontier is
// narrower than a full batch the spare lanes simply idle, which is why the
// scalar path stays as the default.
#[allow(clippy::too_many_arguments)]
fn worker_thread(
    kernel: MuskingumCungeKernel,
    work_rx: Receiver<WorkerMessage>,
    scheduler_tx: Sender<SchedulerMessage>,
    topology: Arc<NetworkTopology>,
    channel_params_map: Arc<FxHashMap<u32, ChannelParams>>,
    max_timesteps: usize,
    dt: f32,
    downsampling: usize,
    writer_tx: Sender<WriterMessage>,
    progress_bar: Arc<ProgressBar>,
    bracket: SecantBracket,
) -> Result<()> {
    let batched = matches!(kernel, MuskingumCungeKernel::RouteRsSimd);
    let mut shutdown_after_batch = false;

    loop {
        let first = match work_rx.recv() {
            Ok(WorkerMessage::ProcessNode(node_id)) => node_id,
            Ok(WorkerMessage::Shutdown) => break,
            Err(e) => {
                eprintln!("Worker channel error: {}", e);
                break;
            }
        };

        let mut node_ids = vec![first];
        if batched {
            while node_ids.len() < LANES {
                match work_rx.try_recv() {
                    Ok(WorkerMessage::ProcessNode(node_id)) => node_ids.push(node_id),
                    Ok(WorkerMessage::Shutdown) => {
                        shutdown_after_batch = true;
                        break;
                    }
                    Err(_) => break,
                }
            }
        }

        // Resolve inputs first; nodes with no forcing at all route to zeros
        // without entering a kernel.
        let mut prepared: Vec<(u32, NodeWork, ChannelParams)> = Vec::with_capacity(node_ids.len());
        let mut routed: Vec<(u32, SimulationResults)> = Vec::with_capacity(node_ids.len());

        for &node_id in &node_ids {
            let Some(params) = channel_params_map.get(&node_id) else {
                continue;
            };
            match prepare_node(&node_id, &topology, max_timesteps) {
                Ok(None) => routed.push((node_id, zero_results(node_id, max_timesteps))),
                Ok(Some(work)) => prepared.push((node_id, work, params.clone())),
                Err(e) => {
                    report_node_error(node_id, &e);
                    writer_tx.send(WriterMessage::Shutdown).ok();
                    scheduler_tx.send(SchedulerMessage::Shutdown).ok();
                }
            }
        }

        if batched {
            for chunk in prepared.chunks(LANES) {
                let results = route_nodes_simd(chunk, max_timesteps, dt, bracket);
                for ((node_id, _, _), r) in chunk.iter().zip(results) {
                    routed.push((*node_id, r));
                }
            }
        } else {
            for (node_id, work, params) in &prepared {
                routed.push((
                    *node_id,
                    route_node_scalar(kernel, *node_id, work, params, max_timesteps, dt, bracket),
                ));
            }
        }

        for (node_id, results) in routed {
            if let Err(e) = finish_node(node_id, results, &topology, downsampling, &writer_tx) {
                report_node_error(node_id, &e);
            }
        }

        // Every node handed to this worker counts as complete, including any
        // skipped for missing channel parameters, or the scheduler stalls.
        for node_id in node_ids {
            if channel_params_map.contains_key(&node_id) {
                progress_bar.inc(1);
            }
            if let Err(e) = scheduler_tx.send(SchedulerMessage::NodeCompleted(node_id)) {
                eprintln!("Failed to notify scheduler of completion: {}", e);
            }
        }

        if shutdown_after_batch {
            break;
        }
    }
    Ok(())
}

// Main parallel routing function
pub fn process_routing_parallel(
    kernel: MuskingumCungeKernel,
    topology: Arc<NetworkTopology>,
    channel_params_map: Arc<FxHashMap<u32, ChannelParams>>,
    max_timesteps: usize,
    dt: f32,
    downsampling: usize,
    output_file: Arc<Mutex<FileMut>>,
    progress_bar: Arc<ProgressBar>,
    num_threads: usize,
    bracket: SecantBracket,
) -> Result<()> {
    let total_nodes = topology.nodes.len();
    let topology_arc = topology;
    let channel_params_arc = channel_params_map;

    // Create channels
    let (writer_tx, writer_rx) = mpsc::channel();
    let (scheduler_tx, scheduler_rx) = mpsc::channel();

    // Create worker channels
    println!(
        "Using {} worker threads for parallel processing",
        num_threads
    );

    let mut worker_txs = Vec::new();
    let mut worker_handles = Vec::new();

    // Spawn worker threads
    for i in 0..num_threads {
        let (work_tx, work_rx) = mpsc::channel();
        worker_txs.push(work_tx);

        let topo = Arc::clone(&topology_arc);
        let params = Arc::clone(&channel_params_arc);
        let writer = writer_tx.clone();
        let scheduler = scheduler_tx.clone();
        let pb = Arc::clone(&progress_bar);

        let handle = thread::spawn(move || {
            if let Err(e) = worker_thread(
                kernel,
                work_rx,
                scheduler,
                topo,
                params,
                max_timesteps,
                dt,
                downsampling,
                writer,
                pb,
                bracket,
            ) {
                eprintln!("Worker {} error: {}", i, e);
            }
        });
        worker_handles.push(handle);
    }

    // Spawn writer thread
    let output_file_clone = Arc::clone(&output_file);
    let writer_handle = thread::spawn(move || {
        if let Err(e) = writer_thread(writer_rx, output_file_clone, min(100, total_nodes)) {
            eprintln!("Writer thread error: {}", e);
        }
    });

    // Spawn scheduler thread
    let topo = Arc::clone(&topology_arc);
    let scheduler_handle = thread::spawn(move || {
        if let Err(e) = scheduler_thread(topo, scheduler_rx, worker_txs) {
            eprintln!("Scheduler thread error: {}", e);
        }
    });

    // Drop original senders
    drop(writer_tx);
    drop(scheduler_tx);

    // Wait for all threads to complete
    scheduler_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Scheduler thread panicked: {:?}", e))?;

    for (i, handle) in worker_handles.into_iter().enumerate() {
        handle
            .join()
            .map_err(|e| anyhow::anyhow!("Worker thread {} panicked: {:?}", i, e))?;
    }

    writer_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Writer thread panicked: {:?}", e))?;

    progress_bar.finish_with_message("Complete");
    println!("Successfully processed all {} nodes", total_nodes);

    Ok(())
}
