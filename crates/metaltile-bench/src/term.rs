use std::{
    io::{IsTerminal, stderr, stdout},
    sync::OnceLock,
};

#[derive(Clone, Copy)]
pub enum Stream {
    Stdout,
    Stderr,
}

#[derive(Clone, Copy)]
pub enum Color {
    Red,
    Green,
    Yellow,
    Cyan,
    BrightBlack,
    BrightWhite,
}

impl Color {
    fn code(self) -> &'static str {
        match self {
            Self::Red => "31",
            Self::Green => "32",
            Self::Yellow => "33",
            Self::Cyan => "36",
            Self::BrightBlack => "90",
            Self::BrightWhite => "97",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Style {
    fg: Option<Color>,
    bold: bool,
    dim: bool,
}

impl Style {
    pub fn new() -> Self { Self::default() }

    pub fn fg(mut self, color: Color) -> Self {
        self.fg = Some(color);
        self
    }

    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    pub fn dim(mut self) -> Self {
        self.dim = true;
        self
    }
}

pub fn paint_stdout(text: impl AsRef<str>, style: Style) -> String {
    paint(Stream::Stdout, text.as_ref(), style)
}

pub fn paint_stderr(text: impl AsRef<str>, style: Style) -> String {
    paint(Stream::Stderr, text.as_ref(), style)
}

fn paint(stream: Stream, text: &str, style: Style) -> String {
    if text.is_empty() || !enabled(stream) {
        return text.to_owned();
    }

    let mut codes = Vec::with_capacity(3);
    if style.bold {
        codes.push("1");
    }
    if style.dim {
        codes.push("2");
    }
    if let Some(color) = style.fg {
        codes.push(color.code());
    }
    if codes.is_empty() {
        return text.to_owned();
    }
    format!("\x1b[{}m{text}\x1b[0m", codes.join(";"))
}

fn enabled(stream: Stream) -> bool {
    static STDOUT_ENABLED: OnceLock<bool> = OnceLock::new();
    static STDERR_ENABLED: OnceLock<bool> = OnceLock::new();

    match stream {
        Stream::Stdout => *STDOUT_ENABLED.get_or_init(|| detect(Stream::Stdout)),
        Stream::Stderr => *STDERR_ENABLED.get_or_init(|| detect(Stream::Stderr)),
    }
}

fn detect(stream: Stream) -> bool {
    if matches!(std::env::var("CLICOLOR_FORCE").as_deref(), Ok(v) if v != "0") {
        return true;
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if matches!(std::env::var("CLICOLOR").as_deref(), Ok("0")) {
        return false;
    }
    if matches!(std::env::var("TERM").as_deref(), Ok("dumb")) {
        return false;
    }

    match stream {
        Stream::Stdout => stdout().is_terminal(),
        Stream::Stderr => stderr().is_terminal(),
    }
}
