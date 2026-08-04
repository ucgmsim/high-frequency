//! Fortran-compatible input: list-directed reads, and the parameter deck.
//!
//! `read(5,*)` is not `split_whitespace().parse()`. The differences that matter
//! here are documented in `PORTING_RULES.md` §8 and are all exercised by the
//! production deck, so none of them are theoretical.

use std::fmt;

/// Errors from reading the deck or an input file.
#[derive(Debug)]
pub enum DeckError {
    /// Ran out of records while a read still wanted items.
    UnexpectedEof { wanted: usize, got: usize },
    /// A token was not a valid number of the requested type.
    BadNumber { token: String, kind: &'static str },
}

impl fmt::Display for DeckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeckError::UnexpectedEof { wanted, got } => write!(
                f,
                "end of input during a list-directed read: wanted {wanted} items, got {got}"
            ),
            DeckError::BadNumber { token, kind } => {
                write!(f, "{token:?} is not a valid {kind}")
            }
        }
    }
}

impl std::error::Error for DeckError {}

/// A reader over records (lines), implementing Fortran list-directed semantics.
///
/// The three behaviours that matter:
///
/// * **A read starts at a fresh record.** Anything left on the previous read's
///   last record is discarded. This is why the third value on the deck's
///   `-1 -1 -1` line never reaches the program.
/// * **A read spans records until its item list is full.** The deck's
///   `ispar_adjust, targ_mag, fault_area` read wants three items and takes one
///   from a bare `0` line and two from the next.
/// * **Blank records are separators, not end of data.** The production deck
///   starts with a blank line and the first read still lands on `sdrop`.
pub struct ListReader {
    records: Vec<String>,
    /// Index of the record the next read will start at.
    next_rec: usize,
}

impl ListReader {
    pub fn new(text: &str) -> Self {
        Self {
            records: text.lines().map(|s| s.to_string()).collect(),
            next_rec: 0,
        }
    }

    /// `read(unit,'(aN)') s` — take the next record verbatim.
    ///
    /// A formatted read does **not** skip blank records: it takes whatever comes
    /// next, blank or otherwise.
    pub fn read_char(&mut self) -> Result<String, DeckError> {
        if self.next_rec >= self.records.len() {
            return Err(DeckError::UnexpectedEof { wanted: 1, got: 0 });
        }
        let s = self.records[self.next_rec].clone();
        self.next_rec += 1;
        Ok(s)
    }

    /// `read(unit,'(aN)')` followed by the trim the main program applies:
    /// `loc = index(name,' ')-1`, i.e. cut at the **first** blank.
    ///
    /// Not "trim trailing whitespace": a path containing a space is truncated at
    /// that space, exactly as the Fortran does.
    pub fn read_filename(&mut self) -> Result<String, DeckError> {
        let s = self.read_char()?;
        Ok(match s.find(' ') {
            Some(i) => s[..i].to_string(),
            None => s,
        })
    }

    /// Collect `n` list-directed tokens, spanning records as needed.
    ///
    /// A `/` terminates the read early; the caller keeps whatever defaults it had
    /// for the unfilled items, which is why this returns a short vector rather
    /// than erroring. A repeat count `r*value` expands to `r` copies. An empty
    /// token (from `,,`) is returned as `None`, meaning "leave unchanged".
    pub fn read_values(&mut self, n: usize) -> Result<Vec<Option<String>>, DeckError> {
        let mut out: Vec<Option<String>> = Vec::with_capacity(n);
        let mut rec = self.next_rec;

        while out.len() < n {
            if rec >= self.records.len() {
                return Err(DeckError::UnexpectedEof { wanted: n, got: out.len() });
            }
            let line = self.records[rec].clone();
            let mut terminated = false;

            for tok in tokenize(&line) {
                match tok {
                    Token::Slash => {
                        terminated = true;
                        break;
                    }
                    Token::Null => out.push(None),
                    Token::Value(v) => out.push(Some(v)),
                    Token::Repeat(count, v) => {
                        for _ in 0..count {
                            out.push(Some(v.clone()));
                            if out.len() == n {
                                break;
                            }
                        }
                    }
                }
                if out.len() >= n {
                    break;
                }
            }

            // The record we just consumed from is finished as far as any later
            // read is concerned.
            rec += 1;
            if terminated {
                break;
            }
        }

        self.next_rec = rec;
        out.truncate(n);
        Ok(out)
    }

    /// `read(unit,*) n, (arr(i), i=1,n)` — one read whose item count depends on
    /// its own first item.
    ///
    /// This is a single Fortran read statement, not two: the implied-do bound is
    /// evaluated after `n` is read, and the whole list may span records. Reading
    /// `n` separately and then the array would apply the fresh-record rule in
    /// between and lose any values that followed `n` on the same line.
    ///
    /// Returns `(n, items)` where `items` has `n` entries.
    pub fn read_count_and_list(&mut self) -> Result<(i32, Vec<Option<String>>), DeckError> {
        let mut all: Vec<Option<String>> = Vec::new();
        let mut want: Option<usize> = None; // total items, known once n is read
        let mut rec = self.next_rec;

        loop {
            if want.is_some_and(|w| all.len() >= w) {
                break;
            }
            if rec >= self.records.len() {
                return Err(DeckError::UnexpectedEof {
                    wanted: want.unwrap_or(1),
                    got: all.len(),
                });
            }
            let line = self.records[rec].clone();
            let mut terminated = false;
            for tok in tokenize(&line) {
                match tok {
                    Token::Slash => { terminated = true; break; }
                    Token::Null => all.push(None),
                    Token::Value(v) => all.push(Some(v)),
                    Token::Repeat(c, v) => {
                        for _ in 0..c { all.push(Some(v.clone())); }
                    }
                }
                if want.is_none() && !all.is_empty() {
                    let n = parse_i32(all[0].as_deref().unwrap_or(""))?;
                    want = Some(1 + n.max(0) as usize);
                }
                if want.is_some_and(|w| all.len() >= w) { break; }
            }
            rec += 1;
            if terminated { break; }
        }

        self.next_rec = rec;
        let n = parse_i32(all.first().and_then(|x| x.as_deref()).unwrap_or(""))?;
        let items = all.into_iter().skip(1).take(n.max(0) as usize).collect();
        Ok((n, items))
    }

    /// One list-directed `f32`.
    pub fn f32(&mut self) -> Result<f32, DeckError> {
        let v = self.read_values(1)?;
        parse_f32(v.first().and_then(|x| x.as_deref()).unwrap_or(""))
    }

    /// One list-directed `f64`.
    pub fn f64(&mut self) -> Result<f64, DeckError> {
        let v = self.read_values(1)?;
        parse_f64(v.first().and_then(|x| x.as_deref()).unwrap_or(""))
    }

    /// One list-directed `i32`.
    pub fn i32(&mut self) -> Result<i32, DeckError> {
        let v = self.read_values(1)?;
        parse_i32(v.first().and_then(|x| x.as_deref()).unwrap_or(""))
    }
}

enum Token {
    Value(String),
    Repeat(usize, String),
    Null,
    Slash,
}

/// Split one record into list-directed tokens.
///
/// Separators are commas and runs of blanks. A comma with nothing before it (or
/// two in a row) is a null value. Everything after a `/` is ignored.
fn tokenize(line: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0usize;
    let mut expect_value_after_comma = false;

    while i < bytes.len() {
        let ch = bytes[i];
        if ch == ' ' || ch == '\t' {
            i += 1;
            continue;
        }
        if ch == ',' {
            // A comma directly after a separator or another comma is a null.
            if expect_value_after_comma {
                out.push(Token::Null);
            }
            expect_value_after_comma = true;
            i += 1;
            continue;
        }
        if ch == '/' {
            out.push(Token::Slash);
            return out;
        }
        // A bare value.
        let start = i;
        while i < bytes.len() && bytes[i] != ' ' && bytes[i] != '\t' && bytes[i] != ',' {
            if bytes[i] == '/' {
                break;
            }
            i += 1;
        }
        let raw: String = bytes[start..i].iter().collect();
        // Repeat count: r*value, where r is a positive integer. Note `3*` with
        // nothing after it is a repeated null, which we do not need to support.
        if let Some((count, value)) = raw.split_once('*')
            && let Ok(c) = count.parse::<usize>()
            && !value.is_empty()
        {
            out.push(Token::Repeat(c, value.to_string()));
            expect_value_after_comma = false;
            continue;
        }
        out.push(Token::Value(raw));
        expect_value_after_comma = false;
    }
    out
}

/// Normalise a Fortran numeric literal for Rust's parser.
///
/// Fortran accepts `D` and `Q` exponent markers, and a bare `1.0+5` form with no
/// marker at all. Rust accepts neither.
fn normalize_number(tok: &str) -> String {
    let t = tok.trim();
    let mut s = String::with_capacity(t.len() + 1);
    let chars: Vec<char> = t.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            'd' | 'D' | 'q' | 'Q' => s.push('e'),
            '+' | '-' if i > 0 => {
                // Exponent with the marker omitted: 1.0+5 means 1.0e+5. Only
                // when the previous character is a digit or a dot, so that a
                // leading sign is untouched.
                let prev = chars[i - 1];
                if prev.is_ascii_digit() || prev == '.' {
                    s.push('e');
                }
                s.push(c);
            }
            _ => s.push(c),
        }
    }
    s
}

pub fn parse_f32(tok: &str) -> Result<f32, DeckError> {
    normalize_number(tok)
        .parse::<f32>()
        .map_err(|_| DeckError::BadNumber { token: tok.to_string(), kind: "real*4" })
}

pub fn parse_f64(tok: &str) -> Result<f64, DeckError> {
    normalize_number(tok)
        .parse::<f64>()
        .map_err(|_| DeckError::BadNumber { token: tok.to_string(), kind: "real*8" })
}

pub fn parse_i32(tok: &str) -> Result<i32, DeckError> {
    let s = normalize_number(tok);
    if let Ok(v) = s.parse::<i32>() {
        return Ok(v);
    }
    // Fortran would reject a real here, but the deck is machine-generated and
    // hf_sim.py can emit "1" or "1.0" depending on the yaml type, so accept a
    // whole real the way a tolerant reader would.
    s.parse::<f64>()
        .map(|v| v as i32)
        .map_err(|_| DeckError::BadNumber { token: tok.to_string(), kind: "integer" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_leading_record_is_skipped() {
        // The production deck's first line is blank and the first read still
        // lands on sdrop.
        let mut r = ListReader::new("\n50.0\nnext\n");
        assert_eq!(r.f32().unwrap(), 50.0);
        assert_eq!(r.read_char().unwrap(), "next");
    }

    #[test]
    fn a_read_spans_records_until_full() {
        // hf_sim.py emits a bare 0, then "-1 -1 -1". One read of three items
        // takes the 0 and then two from the next record.
        let mut r = ListReader::new("0\n-1 -1 -1\n0\n");
        let v = r.read_values(3).unwrap();
        assert_eq!(
            v.iter().map(|x| x.as_deref().unwrap()).collect::<Vec<_>>(),
            ["0", "-1", "-1"]
        );
        // The third -1 is discarded: the next read starts at a fresh record.
        assert_eq!(r.i32().unwrap(), 0);
    }

    #[test]
    fn leftovers_on_a_record_are_discarded() {
        let mut r = ListReader::new("1 2 3\n4\n");
        assert_eq!(r.i32().unwrap(), 1);
        assert_eq!(r.i32().unwrap(), 4, "2 and 3 must be discarded");
    }

    #[test]
    fn slash_terminates_the_read() {
        let mut r = ListReader::new("1 2 / 3 4\n9\n");
        let v = r.read_values(4).unwrap();
        assert_eq!(v.len(), 2, "slash ends the read, leaving items unset");
        assert_eq!(r.i32().unwrap(), 9);
    }

    #[test]
    fn repeat_counts_expand() {
        let mut r = ListReader::new("3*1.5 2.5\n");
        let v = r.read_values(4).unwrap();
        let got: Vec<&str> = v.iter().map(|x| x.as_deref().unwrap()).collect();
        assert_eq!(got, ["1.5", "1.5", "1.5", "2.5"]);
    }

    #[test]
    fn null_values_are_reported_as_none() {
        let mut r = ListReader::new("1,,3\n");
        let v = r.read_values(3).unwrap();
        assert_eq!(v[0].as_deref(), Some("1"));
        assert_eq!(v[1], None, "a doubled comma leaves the item unchanged");
        assert_eq!(v[2].as_deref(), Some("3"));
    }

    #[test]
    fn commas_and_blanks_both_separate() {
        let mut r = ListReader::new("1, 2 ,3   4\n");
        let v = r.read_values(4).unwrap();
        let got: Vec<&str> = v.iter().map(|x| x.as_deref().unwrap()).collect();
        assert_eq!(got, ["1", "2", "3", "4"]);
    }

    #[test]
    fn fortran_exponent_markers() {
        assert_eq!(parse_f64("1.0d+00").unwrap(), 1.0);
        assert_eq!(parse_f64("2.5D-2").unwrap(), 0.025);
        assert_eq!(parse_f32("1.0E5").unwrap(), 1.0e5);
        // Marker omitted entirely, which Fortran accepts and Rust does not.
        assert_eq!(parse_f32("1.5+3").unwrap(), 1500.0);
        assert_eq!(parse_f32("-1.5").unwrap(), -1.5);
        assert_eq!(parse_f32("+2").unwrap(), 2.0);
    }

    #[test]
    fn filename_is_cut_at_the_first_blank() {
        // Not "trim trailing whitespace" -- the Fortran uses index(name,' ')-1,
        // so an embedded space truncates the path.
        let mut r = ListReader::new("/tmp/out.bin   \n/tmp/has space/x\n");
        assert_eq!(r.read_filename().unwrap(), "/tmp/out.bin");
        assert_eq!(r.read_filename().unwrap(), "/tmp/has");
    }

    #[test]
    fn count_and_list_is_one_read() {
        // `read(5,*) nrtyp,(irtype(i),i=1,nrtyp)` with the whole list on one
        // record. Reading nrtyp separately would discard the rest of the line.
        let mut r = ListReader::new("2 1 3\nnext\n");
        let (n, items) = r.read_count_and_list().unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            items.iter().map(|x| x.as_deref().unwrap()).collect::<Vec<_>>(),
            ["1", "3"]
        );
        assert_eq!(r.read_char().unwrap(), "next");
    }

    #[test]
    fn count_and_list_spans_records() {
        let mut r = ListReader::new("3\n7 8\n9\nafter\n");
        let (n, items) = r.read_count_and_list().unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            items.iter().map(|x| x.as_deref().unwrap()).collect::<Vec<_>>(),
            ["7", "8", "9"]
        );
        assert_eq!(r.read_char().unwrap(), "after");
    }

    #[test]
    fn eof_during_a_read_is_an_error() {
        let mut r = ListReader::new("1 2\n");
        assert!(matches!(
            r.read_values(5),
            Err(DeckError::UnexpectedEof { .. })
        ));
    }
}
