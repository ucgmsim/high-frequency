c=======================================================================
c intrinsics_driver.f -- sweep the f32/f64 intrinsics the port relies on.
c
c This is not a kernel gate. It validates the ASSUMPTION underlying every
c other gate: that gfortran's libm and Rust's agree bit for bit on the
c transcendentals this program uses.
c
c PORTING_RULES.md section 10 names this as the likeliest residual source of
c bit-identity failure, alongside ** expansion. Having it as a standalone
c sweep means a future libm divergence shows up here, pointing straight at
c the cause, instead of surfacing as a mysterious mismatch inside stoc_f.
c
c stdin:  output-file
c
c Record: 2 real*4 inputs, then 10 real*4 results, then 2 real*8 inputs and
c 6 real*8 results. See gen_intrinsics_golden.sh for the field order.
c=======================================================================
      program intrinsics_driver

      implicit none
      integer i, seed
      character*256 outfile
      real*4 x, e, urand
      real*8 dx, de, durand

      read(5,'(a256)') outfile

      seed = 20260804
      call init_random_seed(seed)

      open(10, file=outfile(1:index(outfile,' ')-1),
     &     form='unformatted', access='stream', status='replace')

      do i = 1, 20000
c        x spans the magnitudes the program actually sees: frequencies from
c        1e-3 to 1e2 Hz, velocities of order 1, and the very large Rxx
c        (range in cm) that appears in stoc_f's exponentials.
         if (mod(i, 4) .eq. 0) then
            x = urand(1.0e-3, 1.0)
         else if (mod(i, 4) .eq. 1) then
            x = urand(1.0, 100.0)
         else if (mod(i, 4) .eq. 2) then
            x = urand(100.0, 1.0e7)
         else
            x = urand(1.0e-6, 1.0e-3)
         endif
         e = urand(-2.5, 2.5)

         write(10) x, e
c        ** with a real exponent (the 161-occurrence risk), and the two
c        constant exponents stoc_f uses.
         write(10) x**e
         write(10) x**0.5
         write(10) x**(-0.5)
         write(10) sqrt(x)
c        Constant-exponent powers: gfortran folds **(-1.0) to a reciprocal
c        but does NOT fold **0.5 to sqrt. Both are recorded so the Rust
c        mapping for each is pinned rather than assumed.
         write(10) x**(-1.0)
         write(10) 1.0/x
         write(10) alog(x)
         write(10) exp(-x*1.0e-3)
         write(10) sin(e)
         write(10) cos(e)
         write(10) atan2(e, x)
         write(10) sqrt(x*x + e*e)

         dx = durand(1.0d-3, 1.0d3)
         de = durand(-2.5d0, 2.5d0)
         write(10) dx, de
         write(10) dsqrt(dx)
         write(10) dlog(dx)
         write(10) dexp(-dx*1.0d-2)
         write(10) dcos(de)
         write(10) dsin(de)
         write(10) datan2(de, dx)
      enddo

      close(10)
      end

      function urand(lo, hi)
      real*4 urand, lo, hi, rand_numb
      urand = lo + (hi - lo) * rand_numb(0)
      return
      end

      function durand(lo, hi)
      real*8 durand, lo, hi
      real*4 rand_numb
      durand = lo + (hi - lo) * dble(rand_numb(0))
      return
      end
