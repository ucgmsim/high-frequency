c=======================================================================
c tier3_driver.f -- golden generator for pnot and ttime.
c
c Both routines sit downstream of trav, so the driver builds /rays/ and
c /vmod/, calls trav to populate /travel/ and /coff/, and only then calls
c the routine under test -- the same sequence gf_amp_tt uses. The
c POST-TRAV state is what gets dumped, so the Rust test loads it directly
c and the kernel under test is isolated from trav (which has its own gate).
c
c stdin:  mode
c         output-file
c
c mode 1  pnot   rec: ndeep, r r8, th/vp/vs(ndeep) r8, alp/als(ndeep) r4,
c                     p0 r8, t0 r8
c mode 2  ttime  rec: ndeep, n, p0 r8, r r8, th/vp/vs(ndeep) r8,
c                     alp/als(ndeep) r4, nh/nm/it/nup1(n) i4,
c                     p1 r8, t1 r8
c=======================================================================
      program tier3_driver

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
         call gen_pnot()
      else if (mode .eq. 2) then
         call gen_ttime()
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
c Build /vmod/ and /rays/, then run trav to fill /travel/ and /coff/.
c
c kc selects the ray shape:
c   1,2,3  plain upgoing ray of increasing depth (the production shape:
c          nh running ksrc down to krec=2, mode 4 = SH)
c   4      Moho-multiple shape with repeated layers, so trav sets it(i)=1
c          and ttime's interface clamp actually runs
c   5      P mode, which makes ttime consider vp as well as vs
c   6      single segment
c-----------------------------------------------------------------------
      subroutine setup(kc, j0, n, md, hs, hr)
      include 'params.h'
      common/vmod/dpt(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            rho(nlaymax),qp(nlaymax),qs(nlaymax)
      common/rays/nh(1,nlaymax),nm(1,nlaymax),ndeg(nlaymax),nd(nlaymax)
      real*8 dpt, th, vp, vs, rho
      real*4 qp, qs
      integer kc, j0, n, md, k, l, j, ksrc, krec
      real*8 hs, hr, depsum
      real*4 urand, frac

      j0 = 34
      krec = 2

      do k = 1, j0
         frac = float(k - 1) / float(j0 - 1)
         th(k) = dble(urand(0.05, 0.4) + 3.0 * frac)
         vs(k) = dble(0.5 + 4.1 * frac + urand(-0.02, 0.02))
         vp(k) = vs(k) * 1.75d0
         rho(k) = dble(1.81 + 1.5 * frac)
         qs(k) = 50.0 + 150.0 * frac
         qp(k) = 2.0 * qs(k)
      enddo

      md = 4
      if (kc .eq. 5) md = 5

      l = 0
      if (kc .eq. 4) then
         do j = 18, krec, -1
            l = l + 1
            nh(1,l) = j
            nm(1,l) = md
         enddo
         do j = krec, 18
            l = l + 1
            nh(1,l) = j
            nm(1,l) = md
         enddo
      else if (kc .eq. 6) then
         l = 1
         nh(1,1) = krec
         nm(1,1) = md
      else
         ksrc = 6 + 3 * kc
         do j = ksrc, krec, -1
            l = l + 1
            nh(1,l) = j
            nm(1,l) = md
         enddo
      endif
      n = l
      nd(1) = n
      ndeg(1) = 1

      depsum = 0.0d0
      do k = 1, nh(1,1)
         depsum = depsum + th(k)
      enddo
      hs = depsum - 0.5d0 * th(nh(1,1))
      hr = th(1)

      call trav(1, hs, hr)
      return
      end

      subroutine dump_state(ndp)
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
c mode 1: pnot. Ranges chosen so both the immediate-return path
c (dtau/dp >= 0 at the branch cut) and the bisection path are entered.
c-----------------------------------------------------------------------
      subroutine gen_pnot()
      include 'params.h'
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      real*4 alp, als
      integer kc, ic, j0, n, md
      real*8 hs, hr, r, p0, t0, durand

      do kc = 1, 6
         do ic = 1, 12
            call setup(kc, j0, n, md, hs, hr)
c           Epicentral distance from very near to teleseismic-ish: short
c           ranges give dtau/dp < 0 at the cut and force the bisection.
            if (ic .le. 4) then
               r = durand(0.5d0, 5.0d0)
            else if (ic .le. 8) then
               r = durand(5.0d0, 60.0d0)
            else
               r = durand(60.0d0, 400.0d0)
            endif

            call pnot(1, p0, t0, r)

            write(10) ndeep
            write(10) r
            call dump_state(ndeep)
            write(10) p0, t0
         enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 2: ttime.
c-----------------------------------------------------------------------
      subroutine gen_ttime()
      include 'params.h'
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      common/rays/nh(1,nlaymax),nm(1,nlaymax),ndeg(nlaymax),nd(nlaymax)
      common/coff/it(nlaymax),nup1(nlaymax)
      real*4 alp, als
      integer kc, ic, j0, n, md, k
      real*8 hs, hr, r, p0, t0, p1, t1, durand

      do kc = 1, 6
         do ic = 1, 12
            call setup(kc, j0, n, md, hs, hr)
            r = durand(1.0d0, 300.0d0)

c           Feed pnot's own output, as gf_amp_tt does.
            call pnot(1, p0, t0, r)
            call ttime(1, p0, t0, p1, t1, r)

            write(10) ndeep, n
            write(10) p0, r
            call dump_state(ndeep)
            write(10) (nh(1,k), k = 1, n)
            write(10) (nm(1,k), k = 1, n)
            write(10) (it(k), k = 1, n)
            write(10) (nup1(k), k = 1, n)
            write(10) p1, t1
         enddo
      enddo
      return
      end
