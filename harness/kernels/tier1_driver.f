c=======================================================================
c tier1_driver.f -- golden generator for the tier-1 kernels.
c
c These routines communicate through common blocks, so the driver declares
c the blocks itself, populates the input state, calls the routine, and dumps
c both the input state and the emitted state. That makes the golden a
c complete specification of the seam, not just of the return values.
c
c stdin:  mode
c         output-file
c
c mode 1  get_sitefacs
c         rec: j0, nfreq, th(j0) r8, vs(j0) r8, dn(j0) r8, fn(nfreq) r4,
c              an(nfreq) r4 out
c mode 2  trav  (sole writer of /travel/, /coff/, /rmode/)
c         rec: j0, n, ir, ndeg, hs r8, hr r8, th(j0) r8,
c              nh(n), nm(n),
c              love, nup, ndeep, it(n), nup1(n), alp(j0) r4, als(j0) r4
c mode 3  geom_terms
c         rec: j0, n, itype, hs r8, p0 r8, th(j0) r8, vs(j0) r8, qs(j0) r4,
c              nh(n), rp r8 out, qb r4 out
c mode 4  even_dist2
c         rec: nx, nw, 10 r4 args, then nx*nw of (dst,rl,th,ph,zet) r4
c
c All output is access='stream', little-endian on x86-64.
c=======================================================================
      program tier1_driver

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
         call gen_sitefacs()
      else if (mode .eq. 2) then
         call gen_trav()
      else if (mode .eq. 3) then
         call gen_geom_terms()
      else if (mode .eq. 4) then
         call gen_even_dist2()
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
c Build a plausible layered model into /vmod/, shaped like the real
c 34-layer velocity_model fixture: thin slow layers near the surface
c thickening and speeding up with depth.
c-----------------------------------------------------------------------
      subroutine build_vmod(j0)
      include 'params.h'
      common/vmod/dep(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            dn(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dep, th, vp, vs, dn
      real*4 qp, qs
      integer j0, k
      real*4 urand, frac

      dep(1) = 0.0d0
      do k = 1, j0
         frac = float(k - 1) / float(j0 - 1)
         th(k) = dble(urand(0.05, 0.4) + 3.0 * frac)
         vs(k) = dble(0.5 + 4.1 * frac + urand(-0.02, 0.02))
         vp(k) = vs(k) * 1.75d0
         dn(k) = dble(1.81 + 1.5 * frac)
         qs(k) = 50.0 + 150.0 * frac
         qp(k) = 2.0 * qs(k)
         if (k .gt. 1) dep(k) = dep(k-1) + th(k)
      enddo
      return
      end

      subroutine dump_vmod(j0)
      include 'params.h'
      common/vmod/dep(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            dn(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dep, th, vp, vs, dn
      real*4 qp, qs
      integer j0, k
      write(10) (th(k), k = 1, j0)
      write(10) (vs(k), k = 1, j0)
      write(10) (dn(k), k = 1, j0)
      return
      end

c-----------------------------------------------------------------------
c mode 1: get_sitefacs. nfreq = 20 matches nsfac in the main program.
c-----------------------------------------------------------------------
      subroutine gen_sitefacs()
      include 'params.h'
      integer j0, nfreq, k, kc
      integer j0s(4)
      real*4 fn(64), an(64), urand
      data j0s /5, 12, 34, 120/

      do kc = 1, 4
         j0 = j0s(kc)
         nfreq = 20
         call build_vmod(j0)
c        Ascending natural-log frequencies, 0.1 to 25 Hz, as the main
c        program's fn array holds after the :216-218 log transform.
         do k = 1, nfreq
            fn(k) = alog(0.1 * (25.0/0.1)**(float(k-1)/float(nfreq-1)))
         enddo
         call get_sitefacs(j0, nfreq, fn, an)

         write(10) j0, nfreq
         call dump_vmod(j0)
         write(10) (fn(k), k = 1, nfreq)
         write(10) (an(k), k = 1, nfreq)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 2: trav.
c
c Ray shapes follow what gf_amp_tt actually builds: for itype odd the
c segments run nh = ksrc, ksrc-1, ... , krec with krec hardwired to 2 and
c the mode hardwired to 4 (SH). Extra shapes cover the branches production
c does not reach: P and SV modes, the single-segment case that sets
c it(1)=2, a repeated-layer ray that produces reflections (it=1), and
c ndeg < 0 which forces nup = +1.
c-----------------------------------------------------------------------
      subroutine gen_trav()
      include 'params.h'
      common/rays/nh(1,nlaymax),nm(1,nlaymax),ndeg(nlaymax),nd(nlaymax)
      common/travel/alp(nlaymax),als(nlaymax),ndeep,nup
      common/coff/it(nlaymax),nup1(nlaymax)
      common/rmode/love
      common/vmod/dep(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            dn(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dep, th, vp, vs, dn
      real*4 qp, qs
      real*4 alp, als

      integer j0, n, ir, ksrc, krec, md, kc, k, l, j, ndg
      real*8 hs, hr, depsum
      real*4 urand

      ir = 1
      krec = 2

      do kc = 1, 8
         j0 = 34
         if (kc .eq. 8) j0 = 150
         call build_vmod(j0)

         md = 4
         if (kc .eq. 2) md = 5
         if (kc .eq. 3) md = 3
         ndg = 1
         if (kc .eq. 6) ndg = -1

         if (kc .eq. 4) then
c           Single segment: source and receiver in the same layer.
            ksrc = krec
         else
            ksrc = 8 + kc
         endif

         l = 0
         if (kc .eq. 5) then
c           Moho-multiple shape: up to krec, back down to j0-1, up again.
c           Adjacent repeats give it(i) = 1 (reflection).
            do j = 20, krec, -1
               l = l + 1
               nh(ir,l) = j
               nm(ir,l) = md
            enddo
            do j = krec, 20
               l = l + 1
               nh(ir,l) = j
               nm(ir,l) = md
            enddo
         else
            do j = ksrc, krec, -1
               l = l + 1
               nh(ir,l) = j
               nm(ir,l) = md
            enddo
         endif
         n = l
         nd(ir) = n
         ndeg(ir) = ndg

c        Source depth inside the deepest layer the ray touches; receiver at
c        the base of layer 1, as gf_amp_tt sets hr = th(1).
         depsum = 0.0d0
         do k = 1, nh(ir,1)
            depsum = depsum + th(k)
         enddo
         hs = depsum - 0.5d0 * th(nh(ir,1))
         hr = th(1)
         if (kc .eq. 7) hr = th(1) + dble(urand(0.0, 0.5))

         call trav(ir, hs, hr)

         write(10) j0, n, ir, ndeg(ir), hs, hr
         call dump_vmod(j0)
         write(10) (nh(ir,k), k = 1, n)
         write(10) (nm(ir,k), k = 1, n)
         write(10) love, nup, ndeep
         write(10) (it(k), k = 1, n)
         write(10) (nup1(k), k = 1, n)
         write(10) (alp(k), k = 1, j0)
         write(10) (als(k), k = 1, j0)
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 3: geom_terms. Covers odd and even itype, and ray parameters both
c below and above the 1/vs branch point so the 0.999999 sini clamp fires.
c-----------------------------------------------------------------------
      subroutine gen_geom_terms()
      include 'params.h'
      common/rays/nh(1,nlaymax),nm(1,nlaymax),ndeg(nlaymax),nd(nlaymax)
      common/vmod/dep(nlaymax),th(nlaymax),vp(nlaymax),vs(nlaymax),
     &            dn(nlaymax),qp(nlaymax),qs(nlaymax)
      real*8 dep, th, vp, vs, dn
      real*4 qp, qs

      integer j0, n, ir, ksrc, krec, kc, k, l, j, itype
      real*8 hs, p0, rp, depsum
      real*4 qb, urand

      ir = 1
      krec = 2
      j0 = 34

      do kc = 1, 6
         call build_vmod(j0)
         ksrc = 8 + kc
         itype = 1
         if (mod(kc, 2) .eq. 0) itype = 2

         l = 0
         do j = ksrc, krec, -1
            l = l + 1
            nh(ir,l) = j
            nm(ir,l) = 4
         enddo
         n = l
         nd(ir) = n
         ndeg(ir) = 1

         depsum = 0.0d0
         do k = 1, nh(ir,1)
            depsum = depsum + th(k)
         enddo
         hs = depsum - 0.5d0 * th(nh(ir,1))

         if (kc .le. 2) then
c           Well below the branch point.
            p0 = 0.5d0 / vs(nh(ir,1))
         else if (kc .le. 4) then
c           Just below.
            p0 = 0.97d0 / vs(nh(ir,1))
         else
c           Above: sini >= 1 for at least one segment, so the clamp fires.
            p0 = 1.4d0 / vs(nh(ir,1))
         endif

         call geom_terms(hs, p0, itype, rp, qb)

         write(10) j0, n, itype, hs, p0
         write(10) (th(k), k = 1, j0)
         write(10) (vs(k), k = 1, j0)
         write(10) (qs(k), k = 1, j0)
         write(10) (nh(ir,k), k = 1, n)
         write(10) rp, qb
      enddo
      return
      end

c-----------------------------------------------------------------------
c mode 4: even_dist2. Pure argument interface, no common blocks.
c-----------------------------------------------------------------------
      subroutine gen_even_dist2()
      include 'params.h'
      integer nx, nw, kc, i, j
      real*4 rl(nq,np), ph(nq,np), th(nq,np), dst(nq,np), zet(nq,np)
      real*4 xlonq, ylatq, slon, slat, azmq, dipangq, zm, astop, dx, dy
      real*4 urand

      do kc = 1, 5
         if (kc .eq. 1) then
            nx = 2
            nw = 2
         else if (kc .eq. 2) then
            nx = 1
            nw = 1
         else if (kc .eq. 3) then
            nx = 17
            nw = 5
         else if (kc .eq. 4) then
            nx = 40
            nw = 11
         else
            nx = 7
            nw = 3
         endif

         ylatq   = urand(-46.0, -36.0)
         xlonq   = urand(166.0, 178.0)
         slat    = ylatq + urand(-0.4, 0.4)
         slon    = xlonq + urand(-0.4, 0.4)
         azmq    = urand(0.0, 360.0)
         dipangq = urand(20.0, 90.0)
         zm      = urand(0.5, 20.0)
         dx      = urand(0.5, 2.5)
         dy      = urand(0.5, 2.5)
         astop   = 0.5 * float(nx) * dx

         call even_dist2(xlonq, ylatq, slon, slat, azmq, dipangq,
     &                   zm, astop, dx, dy, nx, nw,
     &                   rl, ph, th, dst, zet)

         write(10) nx, nw
         write(10) xlonq, ylatq, slon, slat, azmq, dipangq,
     &             zm, astop, dx, dy
         write(10) ((dst(i,j), rl(i,j), th(i,j), ph(i,j), zet(i,j),
     &              j = 1, nw), i = 1, nx)
      enddo
      return
      end
