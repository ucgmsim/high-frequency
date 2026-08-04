//! `hb_high` driver.
//!
//! Reads the 22-line parameter deck on stdin, the `.stoch` slip model, the 1-D
//! velocity model and the station list; writes `ndata * 3` interleaved `f32` to
//! the named output file and one `(1x,f10.4)` epicentral distance to stderr.
//!
//! Not yet implemented — the deck parser and driver are Phase 2 work, ported
//! last so that failures localise to already-verified kernels.

fn main() -> std::process::ExitCode {
    eprintln!("hb_high: driver not yet ported (Phase 2)");
    std::process::ExitCode::FAILURE
}
