/// Supported DeepGEMM GPU architecture families.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Arch {
    /// NVIDIA Hopper SM90.
    Sm90,
    /// NVIDIA Blackwell SM100.
    Sm100,
}

impl Arch {
    /// Returns the DeepGEMM architecture family for a CUDA compute capability.
    pub const fn from_compute_capability(major: i32, _minor: i32) -> Option<Self> {
        match major {
            9 => Some(Self::Sm90),
            10 => Some(Self::Sm100),
            _ => None,
        }
    }

    /// Returns the canonical short architecture name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm90 => "sm90",
            Self::Sm100 => "sm100",
        }
    }
}
