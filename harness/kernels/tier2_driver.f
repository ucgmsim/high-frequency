c=======================================================================
c tier2_driver.f -- golden generator for the tier-2 kernels.
c
c stdin:  mode
c         output-file
c
c mode 1  cagcon    rec: ndeep, p re/im r8, r r8, th/vp/vs(ndeep) r8,
c                        alp/als(ndeep) r4, out re/im r8
c mode 2  dtdp      same layout as mode 1
c mode 3  highcor_f rec: nf, mf, np2, rdna(nf) r4, cw1 in (2*np2 r4),
c                        cw1 out (2*np2 r4), stdd out (np2 r4)
c mode 4  RADFRQ_lin
c                   rec: 6 r4 angles, nfold, nr, seed, dfr(nfold) r4,
c                        fr1 out r4, rdna(nfold) r4, 8 post-call draws r4
c mode 5  RADV_lin  rec: 5 r4 angles, nfold, nr, dfr(nfold) r4,
c                        rna(nr) r4, rnb(nr) r4, fr1 out r4, rdna(nfold) r4
c
c The eight post-call draws in mode 4 are the point of that record: they pin
c the GENERATOR POSITION after the call, so the golden catches a Rust version
c that computes the right radiation pattern while consuming the wrong number
c of deviates. Values alone would not catch that, and the whole program's
c output depends on the shared stream staying in step.
c=======================================================================
      program tier2_driver

      implicit none
      integer mode, seed
      character*256 outfile

      read(5,*) mode
      read(5,'(a256)') outfile

      seed = 20260804
      call init_random_seed(seed)

      open(10, file=outfile(1:index(outfile,' ')-1),
     &     form='unformatted', access='stream', status='replace')

      if (mode .eq. 1) then
         call gen_cagcon()
      else if (mode .eq. 2) then
         call gen_dtdp()
      else if (mode .eq. 3) then
         call gen_highcor_f()
      else if (mode .eq. 4) then
         call gen_radfrq_lin()
      else if (mode .eq. 5) then
         call gen_radv_lin()
      else
         write(0,*) 'unknown mode ', mode
         stop 1
      endif

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

c-----------------------------------------------------------------------
c Populate /vmod/ and /travel/ for the cagcon/dtdp seam.
c
c alp/als deliberately include negatives, which trav produces via its
c source- and receiver-layer adjustments. That matters because cagcon
c guards on alp(i) > 0 while dtdp guards on alp(i) /= 0, so the two
c routines treat negative multipliers differently and only a golden with
c negatives present will catch a port that conflates them.
c-----------------------------------------------------------------------
      subroutine build_ray_state(ndp, kc)
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs, alp, als
      integer ndp, kc, k
      real*4 urand, frac

      do k = 1, ndp
         frac = float(k - 1) / float(max(ndp - 1, 1))
         th(k) = dble(urand(0.05, 0.4) + 3.0 * frac)
         vs(k) = dble(0.5 + 4.1 * frac + urand(-0.02, 0.02))
         vp(k) = vs(k) * 1.75d0
         rho(k) = dble(1.81 + 1.5 * frac)
         qs(k) = 50.0 + 150.0 * frac
         qp(k) = 2.0 * qs(k)

         if (kc .eq. 1) then
c           S only, all positive: the common production case (mode 4 = SH).
            alp(k) = 0.0
            als(k) = 1.0
         else if (kc .eq. 2) then
c           P only.
            alp(k) = 1.0
            als(k) = 0.0
         else if (kc .eq. 3) then
c           Both modes present.
            alp(k) = float(mod(k, 3))
            als(k) = float(mod(k + 1, 3))
         else if (kc .eq. 4) then
c           Negative multipliers: cagcon skips these, dtdp does not.
            alp(k) = urand(-1.5, 1.5)
            als(k) = urand(-1.5, 1.5)
         else
c           Mixed, with exact zeros to hit both guards' skip paths.
            alp(k) = 0.0
            als(k) = 0.0
            if (mod(k, 2) .eq. 0) als(k) = 2.0
            if (mod(k, 5) .eq. 0) alp(k) = -0.5
         endif
      enddo
      ndeep = ndp
      return
      end

      subroutine dump_ray_state(ndp)
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs, alp, als
      integer ndp, k
      write(10) (th(k), k = 1, ndp)
      write(10) (vp(k), k = 1, ndp)
      write(10) (vs(k), k = 1, ndp)
      write(10) (alp(k), k = 1, ndp)
      write(10) (als(k), k = 1, ndp)
      return
      end

c-----------------------------------------------------------------------
c mode 1: cagcon
c-----------------------------------------------------------------------
      subroutine gen_cagcon()
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs
      integer kc, ic, ndp
      complex*16 p, z, cagcon
      real*8 r, pr, pim, durand

      do kc = 1, 5
         do ic = 1, 40
            ndp = 12 + kc
            call build_ray_state(ndp, kc)
            r = durand(5.0d0, 300.0d0)
c           Ray parameters spanning below, at, and above 1/vs so cr's
c           branch cut is exercised through cagcon.
            if (mod(ic, 4) .eq. 0) then
               pr = durand(1.05d0, 2.0d0) / vs(ndp)
            else
               pr = durand(0.0d0, 0.95d0) / vs(ndp)
            endif
            pim = 0.0d0
            if (mod(ic, 3) .eq. 0) pim = durand(-0.05d0, 0.05d0)
            p = dcmplx(pr, pim)

            z = cagcon(p, 1, r)

            write(10) ndp
            write(10) dreal(p), dimag(p), r
            call dump_ray_state(ndp)
            write(10) dreal(z), dimag(z)
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 2: dtdp
c-----------------------------------------------------------------------
      subroutine gen_dtdp()
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs
      integer kc, ic, ndp
      complex*16 p, z, dtdp
      real*8 r, pr, pim, durand

      do kc = 1, 5
         do ic = 1, 40
            ndp = 12 + kc
            call build_ray_state(ndp, kc)
            r = durand(5.0d0, 300.0d0)
            if (mod(ic, 4) .eq. 0) then
               pr = durand(1.05d0, 2.0d0) / vs(ndp)
            else
               pr = durand(0.0d0, 0.95d0) / vs(ndp)
            endif
            pim = 0.0d0
            if (mod(ic, 3) .eq. 0) pim = durand(-0.05d0, 0.05d0)
            p = dcmplx(pr, pim)

            z = dtdp(p, 1, r)

            write(10) ndp
            write(10) dreal(p), dimag(p), r
            call dump_ray_state(ndp)
            write(10) dreal(z), dimag(z)
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 3: highcor_f. nf/mf follow the main program: nf = np2/2+1,
c mf = np2/2-1 (:1121-1122).
c-----------------------------------------------------------------------
      subroutine gen_highcor_f()
      integer kc, i, np2, nf, mf
      integer ns(4)
      complex*8 cw1(8192)
      real*4 stdd(8192), rdna(8192), rein(8192), imin(8192), urand
      data ns /64, 256, 4096, 8192/

      do kc = 1, 4
         np2 = ns(kc)
         nf = np2 / 2 + 1
         mf = np2 / 2 - 1

c        Signed radiation pattern, as RADFRQ_lin emits (polarity * radvh).
         do i = 1, nf
            rdna(i) = urand(-1.2, 1.2)
         enddo
         do i = 1, np2
            rein(i) = urand(-3.0, 3.0)
            imin(i) = urand(-3.0, 3.0)
            cw1(i) = cmplx(rein(i), imin(i))
         enddo

         call highcor_f(nf, mf, np2, cw1, stdd, rdna)

         write(10) nf, mf, np2
         write(10) (rdna(i), i = 1, nf)
         write(10) (rein(i), imin(i), i = 1, np2)
         write(10) (real(cw1(i)), aimag(cw1(i)), i = 1, np2)
         write(10) (stdd(i), i = 1, np2)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 4: RADFRQ_lin. nr = 1000 matches the main program.
c
c dfr spans below fr1 (0.5), between fr1 and fr2 (2.0), and above, so all
c three del branches are entered -- even though radmin = 1.0 forces del to
c 1.0 in every one of them.
c-----------------------------------------------------------------------
      subroutine gen_radfrq_lin()
      integer kc, i, nfold, nr, sd
      real*4 stra, dipa, raka, pa, thaa, cmp, fr1
      real*4 dfr(4096), rdna(4096), rna(4096), rnb(4096)
      real*4 urand, rand_numb
      real*4 pai
      parameter (pai = 3.1415926)

      do kc = 1, 6
         nfold = 200
         nr = 1000
         sd = 1000003 + kc * 7919

         stra = urand(0.0, 2.0*pai)
         dipa = urand(0.0, 0.5*pai)
         raka = urand(-pai, pai)
         pa   = urand(0.0, 2.0*pai)
         thaa = urand(0.5*pai, pai)
         cmp  = 0.0
         if (mod(kc, 2) .eq. 0) cmp = -90.0 * pai / 180.0

         do i = 1, nfold
            dfr(i) = 0.02 * float(i - 1)
         enddo

c        Reseed so the draw sequence inside the call is reproducible and
c        independent of what the driver consumed building the inputs.
         sd = 1000003 + kc * 7919
         call init_random_seed(sd)

         fr1 = 0.02
         call RADFRQ_lin(stra, dipa, raka, pa, thaa, dfr, nfold,
     &                   cmp, fr1, rna, rnb, nr, rdna)

         write(10) stra, dipa, raka, pa, thaa, cmp
         write(10) nfold, nr, 1000003 + kc * 7919
         write(10) (dfr(i), i = 1, nfold)
         write(10) fr1
         write(10) (rdna(i), i = 1, nfold)
c        Generator position after the call.
         write(10) (rand_numb(0), i = 1, 8)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 5: RADV_lin. Consumes no draws from the shared stream -- its random
c numbers come from the caller's rna/rnb, which RANU2 fills once per run.
c
c dfr spans below fr1 (0.001), between fr1 and fr2 (0.01), and above.
c-----------------------------------------------------------------------
      subroutine gen_radv_lin()
      integer kc, i, nfold, nr
      real*4 stra, dipa, raka, pa, thaa, fr1
      real*4 dfr(4096), rdna(4096), rna(4096), rnb(4096)
      real*4 urand
      real*4 pai
      parameter (pai = 3.1415926)

      do kc = 1, 6
         nfold = 200
         nr = 1000

         stra = urand(0.0, 2.0*pai)
         dipa = urand(0.0, 0.5*pai)
         raka = urand(-pai, pai)
         pa   = urand(0.0, 2.0*pai)
         thaa = urand(0.5*pai, pai)

c        Log-spaced from 1e-4 to 20 Hz so the 0.001 and 0.01 breakpoints
c        both fall inside the range.
         do i = 1, nfold
            dfr(i) = 1.0e-4 * (20.0/1.0e-4)**(float(i-1)/float(nfold-1))
         enddo
         call RANU2(nr, rna)
         call RANU2(nr, rnb)

         fr1 = 0.02
         call RADV_lin(stra, dipa, raka, pa, thaa, dfr, nfold,
     &                 fr1, rna, rnb, nr, rdna)

         write(10) stra, dipa, raka, pa, thaa
         write(10) nfold, nr
         write(10) (dfr(i), i = 1, nfold)
         write(10) (rna(i), i = 1, nr)
         write(10) (rnb(i), i = 1, nr)
         write(10) fr1
         write(10) (rdna(i), i = 1, nfold)
      enddo
      return
      end
