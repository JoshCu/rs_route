use anyhow::{Context, Result};
use csv::{ByteRecord, ReaderBuilder, Writer, WriterBuilder};
use std::collections::VecDeque;
use std::fs::File;
use std::io::ErrorKind;
use std::path::PathBuf;

// Function to load external flows for a specific nexus/catchment
//
// The file is read into memory in one syscall and parsed as `ByteRecord`s.
// Both matter: the default `Reader` refills an 8 KiB buffer per chunk, and
// `StringRecord` re-validates every field as UTF-8 even though only one
// column is ever used. Together they cost ~4x the parse time of this path
// on a typical 8760-row ngen output. Records still go through the `csv`
// crate, so quoting and escaping are handled as before.
pub fn load_external_flows(
    csv_file: PathBuf,
    id: &u32,
    var_name: Option<&str>,
    area: f32,
) -> Result<VecDeque<f32>> {
    let bytes = match std::fs::read(&csv_file) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            println!(
                "No external flow file found for {}: {}",
                id,
                csv_file.display()
            );
            return Ok(VecDeque::new());
        }
        Err(err) => {
            return Err(err)
                .with_context(|| format!("Failed to open CSV file: {}", csv_file.display()));
        }
    };

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .flexible(true)
        // Only the headers need blanket trimming; the one data field we read
        // is trimmed individually below, so per-field trimming is wasted work.
        .trim(csv::Trim::Headers)
        .from_reader(&bytes[..]);

    let qlat_index = match var_name {
        Some(var_name) => {
            let headers = rdr.byte_headers().context("Failed to read CSV headers")?;
            headers
                .iter()
                .position(|h| h == var_name.as_bytes())
                .unwrap_or(2)
        }
        None => 2,
    };

    // ngen writes one row per output timestep; most runs are hourly.
    let mut external_flows = Vec::with_capacity(8_760);

    let mut record = ByteRecord::new();
    let mut i = 0;
    while rdr
        .read_byte_record(&mut record)
        .with_context(|| format!("Failed to read record {} in file {}", i, csv_file.display()))?
    {
        let ql_bytes = record
            .get(qlat_index)
            .ok_or_else(|| anyhow::anyhow!("Missing column {} in record {}", qlat_index, i))?;

        let ql_str = std::str::from_utf8(ql_bytes)
            .with_context(|| format!("Non-UTF-8 flow value in record {}", i))?
            .trim();

        let ql = ql_str
            .parse::<f32>()
            .with_context(|| format!("Failed to parse flow value '{}' in record {}", ql_str, i))?;

        // https://github.com/CIROH-UA/ngen/blob/ed2a903730467fa631716c033b757c3dff5fa2bb/include/core/Layer.hpp#L142
        external_flows.push((ql * (area * 1_000_000.0)) / 3600.0);
        i += 1;
    }

    Ok(VecDeque::from(external_flows))
}

// Create CSV writer with headers
pub fn create_csv_writer(path: &str) -> Result<Writer<File>> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("Failed to create CSV writer at {}", path))?;

    // Write header
    wtr.write_record(&["step", "feature_id", "flow", "velocity", "depth"])
        .context("Failed to write CSV header")?;

    Ok(wtr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The previous implementation, kept verbatim as a test oracle so the
    /// faster reader can be shown to produce identical values.
    fn load_external_flows_reference(
        csv_file: PathBuf,
        _id: &u32,
        var_name: Option<&str>,
        area: f32,
    ) -> Result<VecDeque<f32>> {
        let mut external_flows = Vec::new();
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b',')
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(&csv_file)?;

        let qlat_index = match var_name {
            Some(var_name) => {
                let headers = rdr.headers()?;
                headers.iter().position(|h| h == var_name).unwrap_or(2)
            }
            None => 2,
        };

        let mut record = csv::StringRecord::new();
        while rdr.read_record(&mut record)? {
            let ql = record.get(qlat_index).unwrap().trim().parse::<f32>()?;
            external_flows.push((ql * (area * 1_000_000.0)) / 3600.0);
        }
        Ok(VecDeque::from(external_flows))
    }

    fn assert_matches_reference(path: &str, var_name: Option<&str>, area: f32) {
        let want =
            load_external_flows_reference(PathBuf::from(path), &486888, var_name, area).unwrap();
        let got = load_external_flows(PathBuf::from(path), &486888, var_name, area).unwrap();
        assert_eq!(want.len(), got.len(), "row count differs for {}", path);
        assert!(!got.is_empty(), "{} produced no rows", path);
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "row {} differs in {}", i, path);
        }
    }

    /// A catchment file with a real header row, read by column name.
    #[test]
    fn test_matches_reference_named_column() {
        assert_matches_reference(
            "./tests/one_cat/outputs/ngen/cat-486888.csv",
            Some("Q_OUT"),
            14.710_95,
        );
    }

    /// A nexus file with no header and space-padded fields, so the column
    /// lookup falls back to index 2 and per-field trimming matters.
    #[test]
    fn test_matches_reference_headerless_padded() {
        assert_matches_reference(
            "./tests/one_cat/outputs/ngen/nex-486889_long_output.csv",
            Some("Q_OUT"),
            14.710_95,
        );
    }

    /// A missing file is reported, not an error, and yields no flows.
    #[test]
    fn test_missing_file_returns_empty() {
        let got = load_external_flows(
            PathBuf::from("./tests/one_cat/outputs/ngen/does-not-exist.csv"),
            &1,
            Some("Q_OUT"),
            1.0,
        )
        .unwrap();
        assert!(got.is_empty());
    }
}
