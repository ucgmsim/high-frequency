c=======================================================================
c pcg32.f -- deterministic replacement for gfortran's intrinsic RNG.
c
c cREF: This file does not exist in EMOD3D. It replaces the
c cREF: random_number/random_seed pair used by hb_high_v6.0.3.f so that the
c cREF: random stream is reproducible in Rust. See reference/PROVENANCE.md.
c
c PCG32 (O'Neill 2014), "pcg32_random_r" variant: a 64-bit LCG whose output
c is permuted by an xorshift plus a data-dependent rotation. Chosen because
c it is ~15 lines of wrapping multiply-add and shifts, so the Fortran and the
c Rust (crates/hb_high/src/rng.rs) can be diffed line by line.
c
c The Fortran and Rust implementations MUST agree bit for bit. Two details
c carry that guarantee:
c
c   * ISHFT is a LOGICAL shift in Fortran (zero fill), matching Rust's >> on
c     unsigned types. Never substitute an arithmetic shift.
c   * rand_numb takes the top 24 bits and divides by 2**24. Both the integer
c     -> real*4 conversion and the division by a power of two are exact, so
c     the result carries no rounding at all and lands in [0, 1-2**-24].
c     Dividing a full 32-bit value by 2**32 would round, and values near 1
c     would round UP to exactly 1.0, breaking the [0,1) contract that
c     normal_random_number's zero-rejection loops assume.
c
c Requires -fwrapv: the LCG step relies on wrapping signed 64-bit multiply.
c=======================================================================

c-----------------------------------------------------------------------
c Generator state. A common block rather than a module so the file stays
c compilable as fixed-form F77 alongside the rest of the program.
c-----------------------------------------------------------------------
      block data pcg32_init
      integer*8 pcg_state, pcg_inc
      common /pcg32/ pcg_state, pcg_inc
      data pcg_state /0_8/
      data pcg_inc /1442695040888963407_8/
      end

c-----------------------------------------------------------------------
c init_random_seed -- drop-in replacement for the original at line 4062.
c
c The original built a seed array of SEED_WORDS values (irand, irand+1, ...)
c and handed it to random_seed(put=). Critically it also INCREMENTED ITS
c ARGUMENT once per word, and hb_high reads that mutated value at line 1366
c (if(irand.gt.0)) to decide whether to apply rupture-time jitter. So the
c number of seed words is not an implementation detail -- it changes which
c branch the program takes.
c
c SEED_WORDS is therefore pinned to 8, which is what gfortran 16.1.1 reports
c from random_seed(size=n) on x86-64. That keeps the line-1366 branch
c behaving exactly as the production build does. If a future toolchain
c reports a different size, production changes and this constant does not --
c which is the whole point of pinning it here.
c-----------------------------------------------------------------------
      subroutine init_random_seed(irand)

      integer irand
      integer i
      integer*8 pcg_state, pcg_inc
      common /pcg32/ pcg_state, pcg_inc

      integer seed_words
      parameter (seed_words = 8)
      integer*8 pcg_mult
      parameter (pcg_mult = 6364136223846793005_8)

c     Absorb the same arithmetic sequence the original fed to random_seed,
c     and leave irand incremented by exactly seed_words as the original did.
      pcg_state = 0
      pcg_inc = 1442695040888963407_8
      do i = 1, seed_words
         pcg_state = pcg_state * pcg_mult + int(irand, 8)
         irand = irand + 1
      enddo

c     Two discarded draws so the first returned value does not expose the
c     low-quality top bits of a freshly seeded LCG state.
      call pcg32_skip()
      call pcg32_skip()

      return
      end

c-----------------------------------------------------------------------
c pcg32_next -- advance the state and return the permuted 32-bit output as
c a non-negative integer*8 in [0, 2**32).
c-----------------------------------------------------------------------
      function pcg32_next()

      integer*8 pcg32_next
      integer*8 pcg_state, pcg_inc
      common /pcg32/ pcg_state, pcg_inc

      integer*8 pcg_mult
      parameter (pcg_mult = 6364136223846793005_8)
      integer*8 mask32
      parameter (mask32 = 4294967295_8)
      integer*8 old, xorshifted, rot

      old = pcg_state
      pcg_state = old * pcg_mult + pcg_inc

c     xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
      xorshifted = iand(ishft(ieor(ishft(old, -18), old), -27), mask32)
c     rot = old >> 59   (0..31)
      rot = ishft(old, -59)

c     32-bit rotate right by rot. When rot is 0 the second term shifts the
c     whole word out and the mask zeroes it, leaving xorshifted unchanged --
c     which is what the canonical (-rot) & 31 formulation also gives.
      pcg32_next = iand(ior(ishft(xorshifted, -rot),
     &                      ishft(xorshifted, 32 - rot)), mask32)

      return
      end

c-----------------------------------------------------------------------
c pcg32_skip -- advance the state, discarding the output.
c-----------------------------------------------------------------------
      subroutine pcg32_skip()
      integer*8 pcg32_next, dummy
      dummy = pcg32_next()
      return
      end

c-----------------------------------------------------------------------
c rand_numb -- drop-in replacement for the original at line 4136.
c
c Returns real*4 in [0, 1-2**-24]. Exact: 24 bits converted to real*4 is
c lossless and the divisor is a power of two.
c
c The original reseeded when iflag > 0. No live call site does that (every
c one passes the literal 0; only the dead RANN2 passed a variable), but the
c branch is kept so the two sources stay line-comparable.
c-----------------------------------------------------------------------
      function rand_numb(iflag)

      integer iflag
      real*4 rand_numb
      integer*8 pcg32_next

      if (iflag .gt. 0) call init_random_seed(iflag)

      rand_numb = real(ishft(pcg32_next(), -8), 4) / 16777216.0

      return
      end
