c=======================================================================
c tier4_driver.f -- golden generator for stoc_f and gf_amp_tt.
c
c stdin:  mode
c         output-file
c
c mode 1  stoc_f
c    rec: np2, seed, 17 real*4 args (r..bigC order below), dfr(nf) r4,
c         cw out (2*np2 r4), then 8 post-call draws r4
c mode 2  gf_amp_tt
c    rec: j0, itype, md, src_depth r4, range r4,
c         th/vp/vs(j0) r8, qs(j0) r4,
c         nd, nh(nd) i4, nm(nd) i4, ndeep, love,
c         rp0, stime, rpath, qbar (4 r4)
c
c stoc_f draws from the shared stream via normal_random_number, so its record
c also carries the generator position afterwards -- same reasoning as
c RADFRQ_lin in tier 2.
c=======================================================================
      program tier4_driver
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
         call gen_stoc_f()
      else if (mode .eq. 2) then
         call gen_gf_amp_tt()
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

c-----------------------------------------------------------------------
c mode 1: stoc_f.
c
c Parameter values follow the production deck: fmx=10, akapp=0.045,
c qfexp=0.6, Czero=2.1 (bigC), and the time-window shape eps/eta that
c VERSION1 uses. akapp <= 0 is also exercised, since that selects the
c other high-cut branch (the **(-1.0) form rather than exp(-pi*f*kappa)).
c-----------------------------------------------------------------------
      subroutine gen_stoc_f()
      integer kc, i, np2, nf, sd
      integer ns(4)
      complex*8 cw(16384)
      real*4 dfr(16384)
      real*4 r, tw, eps, eta, betvs, row, dt, smt, dlm, fc, fmx, akapp
      real*4 qb, qfe, bigC, df, urand, rand_numb
      data ns /64, 256, 2048, 8192/

      do kc = 1, 4
         np2 = ns(kc)
         nf = np2/2 + 1
         dt = 0.005
         df = 1.0/(np2*dt)
         do i = 1, nf
            dfr(i) = df*float(i-1)
         enddo

         r     = urand(2.0, 150.0)
         tw    = urand(0.5, 8.0)
         eps   = 0.2
         eta   = 0.2
         betvs = urand(2.0, 3.8)
         row   = urand(2.2, 2.9)
         smt   = urand(1.0e20, 1.0e23)
         dlm   = 0.0
         fc    = urand(0.1, 3.0)
         fmx   = 10.0
         akapp = 0.045
         if (kc .eq. 3) akapp = -1.0
         qb    = urand(0.005, 0.05)
         qfe   = 0.6
         bigC  = 2.1

         sd = 4000037 + kc*104729
         call init_random_seed(sd)

         call stoc_f(np2, r, tw, eps, eta, betvs, row, dt, smt, dlm,
     &               fc, fmx, akapp, cw, dfr, qb, qfe, bigC)

         write(10) np2, 4000037 + kc*104729
         write(10) r, tw, eps, eta, betvs, row, dt, smt, dlm,
     &             fc, fmx, akapp, qb, qfe, bigC
         write(10) (dfr(i), i = 1, nf)
         write(10) (real(cw(i)), aimag(cw(i)), i = 1, np2)
         write(10) (rand_numb(0), i = 1, 8)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 2: gf_amp_tt.
c
c The velocity model ends with a zero-thickness layer, as the main program
c produces after Moho truncation, so the Moho-detection test
c (th(j+1) == 0) is live for the even-itype shapes.
c-----------------------------------------------------------------------
      subroutine gen_gf_amp_tt()
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      common/rays/nh(1,nlaymax),nm(1,nlaymax),ndeg(nlaymax),nd(nlaymax)
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      common/rmode/love
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs, alp, als

      integer kc, ic, j0, k, itype, md
      real*4 src_depth, range, rp0, stime, rpath, qbar, urand, frac
      complex sgc
      real*8 depsum

      sgc = cmplx(0.0, 0.0)

      do kc = 1, 8
         j0 = 34
         do k = 1, j0
            frac = float(k - 1) / float(j0 - 1)
            th(k) = dble(urand(0.05, 0.4) + 3.0 * frac)
            vs(k) = dble(0.5 + 4.1 * frac + urand(-0.02, 0.02))
            vp(k) = vs(k) * 1.75d0
            rho(k) = dble(1.81 + 1.5 * frac)
            qs(k) = 50.0 + 150.0 * frac
            qp(k) = 2.0 * qs(k)
         enddo
c        Zero-thickness bottom layer marks the Moho, as the main program's
c        truncation at :332-337 leaves it.
         th(j0) = 0.0d0

         depsum = 0.0d0
         do k = 1, j0 - 1
            depsum = depsum + th(k)
         enddo

         md = 4
         if (kc .eq. 7) md = 5
         if (kc .eq. 8) md = 3
         itype = 1
         if (kc .eq. 4) itype = 2
         if (kc .eq. 5) itype = 3
         if (kc .eq. 6) itype = 4

         do ic = 1, 6
c           Source depths from very shallow to just above the Moho.
            src_depth = real(depsum) * (0.05 + 0.15 * float(ic - 1))
            range = urand(1.0, 250.0)

            call gf_amp_tt(j0, src_depth, range, sgc, itype, md,
     &                     rp0, stime, rpath, qbar)

            write(10) j0, itype, md
            write(10) src_depth, range
            write(10) (th(k), k = 1, j0)
            write(10) (vp(k), k = 1, j0)
            write(10) (vs(k), k = 1, j0)
            write(10) (qs(k), k = 1, j0)
            write(10) nd(1)
            write(10) (nh(1,k), k = 1, nd(1))
            write(10) (nm(1,k), k = 1, nd(1))
            write(10) ndeep, love
            write(10) rp0, stime, rpath, qbar
         enddo
      enddo
      return
      end
