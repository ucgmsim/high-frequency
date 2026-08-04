c=======================================================================
c tier0_driver.f -- golden generator for the tier-0 leaf kernels.
c
c Links against reference/hb_high_subs.f and reference/pcg32.f, so goldens
c come from the same Fortran the oracle binary runs.
c
c Inputs are generated here (using the already bit-verified PCG32) and
c written to the golden file ALONGSIDE the outputs. The Rust test reads the
c inputs from the file rather than regenerating them, so a kernel test can
c never silently pass by comparing two different input sets.
c
c stdin:  mode
c         output-file
c
c mode 1  RDATN     record: 5 real*4 in,  2 real*4 out
c mode 2  DELAZ5    record: 4 real*4 in, 1 integer*4 in, 7 real*4 out
c mode 3  DGAMM     record: 1 real*8 in,  1 real*8 out
c mode 4  cr        record: 3 real*8 in,  2 real*8 out
c mode 5  FLZERO    record: 1 int, 1 real*4 dt, n real*4 in, n real*4 out
c mode 6  FAST      record: 2 int (nnn,ind), 2n real*4 in, 2n real*4 out
c mode 7  siteamp   record: 2 int (np2,nn), then dfr(np2/2), fn(nn), an(nn),
c                           2*np2 real*4 cw in, 2*np2 real*4 cw out
c
c All output is access='stream', little-endian on x86-64.
c=======================================================================
      program tier0_driver

      implicit none

      integer mode
      character*256 outfile
      integer seed

      read(5,*) mode
      read(5,'(a256)') outfile

c     Fixed seed: the input sets must be reproducible across regenerations,
c     otherwise a golden refresh silently changes what is being tested.
      seed = 20260804
      call init_random_seed(seed)

      open(10, file=outfile(1:index(outfile,' ')-1),
     &     form='unformatted', access='stream', status='replace')

      if (mode .eq. 1) then
         call gen_rdatn()
      else if (mode .eq. 2) then
         call gen_delaz5()
      else if (mode .eq. 3) then
         call gen_dgamm()
      else if (mode .eq. 4) then
         call gen_cr()
      else if (mode .eq. 5) then
         call gen_flzero()
      else if (mode .eq. 6) then
         call gen_fast()
      else if (mode .eq. 7) then
         call gen_siteamp()
      else
         write(0,*) 'unknown mode ', mode
         stop 1
      endif

      close(10)
      end

c-----------------------------------------------------------------------
c Uniform deviate in [lo,hi) from the shared PCG32 stream.
c-----------------------------------------------------------------------
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

c-----------------------------------------------------------------------
c mode 1: RDATN over the full angular ranges the program can produce.
c-----------------------------------------------------------------------
      subroutine gen_rdatn()
      integer i
      real*4 str, dip, rak, az, th, rdsh, rdsv, urand
      real*4 pai
      parameter (pai = 3.1415926)

      do i = 1, 2000
         str = urand(0.0, 2.0*pai)
         dip = urand(0.0, 0.5*pai)
         rak = urand(-pai, pai)
         az  = urand(0.0, 2.0*pai)
         th  = urand(0.0, pai)
         call RDATN(str, dip, rak, az, th, rdsh, rdsv)
         write(10) str, dip, rak, az, th, rdsh, rdsv
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 2: DELAZ5. Deliberately covers all three separation regimes, since
c the C1-0.94 / C1+0.94 arithmetic IFs select between numerically distinct
c formulations and random lat/lon pairs would almost never hit the near-0
c or near-180 degree branches.
c-----------------------------------------------------------------------
      subroutine gen_delaz5()
      integer i, k, iflag
      real*4 thei, alei, thsi, alsi, urand
      real*4 delt, deltdg, deltkm, azes, azesdg, azse, azsedg
      real*4 eps

      do k = 1, 4
         do i = 1, 500
            thei = urand(-89.0, 89.0)
            alei = urand(-179.0, 179.0)

            if (k .eq. 1) then
c              Mid-range separation: the well-conditioned label-29 branch.
               thsi = urand(-89.0, 89.0)
               alsi = urand(-179.0, 179.0)
            else if (k .eq. 2) then
c              Nearly coincident: label 31.
               eps = urand(0.0001, 0.05)
               thsi = thei + eps
               alsi = alei + eps
            else if (k .eq. 3) then
c              Nearly antipodal: label 28.
               eps = urand(0.0001, 0.05)
               thsi = -thei + eps
               alsi = alei + 180.0 - eps
               if (alsi .gt. 180.0) alsi = alsi - 360.0
            else
c              Small but not tiny separation, straddling the 0.94 threshold.
               eps = urand(1.0, 25.0)
               thsi = thei + eps
               alsi = alei + eps
            endif

c           iflag <= 0 selects geographic degrees, iflag > 0 geocentric
c           radians. ONLY THE DEGREES PATH IS REACHABLE: even_dist2 sets
c           i=0 immediately before its first DELAZ5 call (:2626) and passes
c           the literal 0 at its second (:2658). The geocentric branch is
c           dead code, so it is deliberately not covered here.
            iflag = 0
            call DELAZ5(thei, alei, thsi, alsi, delt, deltdg, deltkm,
     &                  azes, azesdg, azse, azsedg, iflag)
            write(10) thei, alei, thsi, alsi, iflag,
     &                delt, deltdg, deltkm, azes, azesdg, azse, azsedg
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 3: DGAMM. Covers the three argument-reduction paths (>1.5, [0.5,1.5],
c <0.5 including negatives) plus both error returns.
c-----------------------------------------------------------------------
      subroutine gen_dgamm()
      integer i, k
      real*8 x, y, DGAMM, durand

      do k = 1, 4
         do i = 1, 250
            if (k .eq. 1) then
               x = durand(0.5d0, 1.5d0)
            else if (k .eq. 2) then
               x = durand(1.5d0, 56.0d0)
            else if (k .eq. 3) then
               x = durand(0.001d0, 0.5d0)
            else
c              Negative non-integers: the MG/IABS reduction path.
               x = durand(-6.0d0, -0.001d0)
               if (x .eq. dble(int(x))) x = x + 0.25d0
            endif
            y = DGAMM(x)
            write(10) x, y
         enddo
      enddo

c     Error paths. DGAMM writes a diagnostic to unit 6 and returns 1.0D75.
      x = 58.0d0
      y = DGAMM(x)
      write(10) x, y
      x = 0.0d0
      y = DGAMM(x)
      write(10) x, y
      x = -3.0d0
      y = DGAMM(x)
      write(10) x, y
      return
      end

c-----------------------------------------------------------------------
c mode 4: cr. The branch-cut selection makes coverage of the special cases
c essential: |Im p| below 1e-8 with a<0 and a>0, and the f<=t1/e>0 case
c where the sign is NOT flipped.
c-----------------------------------------------------------------------
      subroutine gen_cr()
      integer i, k
      complex*16 p, z, cr
      real*8 v, pr, pim, durand

      do k = 1, 5
         do i = 1, 300
            v = durand(0.5d0, 8.0d0)
            if (k .eq. 1) then
c              General complex ray parameter.
               pr  = durand(-0.5d0, 0.5d0)
               pim = durand(-0.5d0, 0.5d0)
            else if (k .eq. 2) then
c              Purely real, below the branch point: a > 0, phi = 0.
               pr  = durand(0.0d0, 0.9d0) / v
               pim = 0.0d0
            else if (k .eq. 3) then
c              Purely real, above the branch point: a < 0, phi = pi.
               pr  = durand(1.1d0, 3.0d0) / v
               pim = 0.0d0
            else if (k .eq. 4) then
c              Imaginary part just below the 1e-8 threshold.
               pr  = durand(0.0d0, 2.0d0) / v
               pim = durand(-9.0d-9, 9.0d-9)
            else
c              Imaginary part just above the threshold.
               pr  = durand(0.0d0, 2.0d0) / v
               pim = durand(1.1d-8, 1.0d-6)
            endif
            p = dcmplx(pr, pim)
            z = cr(p, v)
            write(10) dreal(p), dimag(p), v, dreal(z), dimag(z)
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 5: FLZERO at several lengths. Only A(3:N) is modified, so the golden
c also pins that A(1) and A(2) come back untouched.
c-----------------------------------------------------------------------
      subroutine gen_flzero()
      integer i, k, n
      integer ns(6)
      real*4 a(8192), ain(8192), dt, urand
      data ns /3, 4, 16, 100, 4096, 8192/

      do k = 1, 6
         n = ns(k)
         dt = urand(0.001, 0.02)
         do i = 1, n
            ain(i) = urand(-20.0, 20.0)
            a(i) = ain(i)
         enddo
         call FLZERO(n, dt, a)
         write(10) n, dt
         write(10) (ain(i), i = 1, n)
         write(10) (a(i), i = 1, n)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 6: FAST forward and inverse at every power of two the program can
c produce, from the degenerate n=1 up to 4096.
c-----------------------------------------------------------------------
      subroutine gen_fast()
      integer i, k, nnn, ind, idir
      complex*8 ace(4096)
      real*4 re(4096), im(4096), urand

      do idir = 1, 2
         ind = -1
         if (idir .eq. 2) ind = 1
         nnn = 1
         do k = 1, 13
            do i = 1, nnn
               re(i) = urand(-5.0, 5.0)
               im(i) = urand(-5.0, 5.0)
               ace(i) = cmplx(re(i), im(i))
            enddo
            call FAST(nnn, ace, ind)
            write(10) nnn, ind
            write(10) (re(i), im(i), i = 1, nnn)
            write(10) (real(ace(i)), aimag(ace(i)), i = 1, nnn)
            nnn = nnn * 2
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 7: siteamp. Builds a plausible frequency axis and an ascending
c log-frequency amplification table, matching what get_sitefacs produces
c (nsfac = 20 in the main program).
c-----------------------------------------------------------------------
      subroutine gen_siteamp()
      integer i, k, np2, nn, np
      integer ns(4)
      complex*8 cw(8192)
      real*4 dfr(8192), fn(64), an(64)
      real*4 rein(8192), imin(8192)
      real*4 df, dt, urand
      data ns /64, 256, 4096, 8192/

      do k = 1, 4
         np2 = ns(k)
         np = np2 / 2
         nn = 20
         dt = 0.005
         df = 1.0 / (np2 * dt)

c        dfr(1) = 0, matching the main program's axis at :1124. siteamp
c        starts its loop at i=2 precisely because alog(0) is undefined.
         do i = 1, np + 1
            dfr(i) = df * float(i - 1)
         enddo

c        Ascending natural-log frequencies from 0.1 to 25 Hz, with log
c        amplifications, as get_sitefacs emits.
         do i = 1, nn
            fn(i) = alog(0.1 * (25.0/0.1)**(float(i-1)/float(nn-1)))
            an(i) = urand(-0.5, 1.5)
         enddo

         do i = 1, np2
            rein(i) = urand(-3.0, 3.0)
            imin(i) = urand(-3.0, 3.0)
            cw(i) = cmplx(rein(i), imin(i))
         enddo

         call siteamp(np2, cw, dfr, nn, fn, an)

         write(10) np2, nn
         write(10) (dfr(i), i = 1, np + 1)
         write(10) (fn(i), i = 1, nn)
         write(10) (an(i), i = 1, nn)
         write(10) (rein(i), imin(i), i = 1, np2)
         write(10) (real(cw(i)), aimag(cw(i)), i = 1, np2)
      enddo
      return
      end
