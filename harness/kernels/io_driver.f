c=======================================================================
c io_driver.f -- golden generator for the input readers.
c
c The read blocks below are copied VERBATIM from hb_high_orig.f so the golden
c reflects what the program actually parses, not what the port's author
c believes it parses. Line references are to the vendored original.
c
c Unit tests in input.rs check the port against my reading of the format;
c this checks it against the Fortran.
c
c stdin:  stoch-file
c         velmod-file
c         station-file
c         nsite
c         output-file
c=======================================================================
      program io_driver
      include 'params.h'
c     No `implicit none`: params.h's PARAMETER names rely on implicit typing,
c     exactly as in the main program. Everything below is declared anyway.

      character*256 slip_model, velfile, asite, outfile, dummy
      integer nevnt, iv, i, j, j0, jmoho, nstot, nsite, msite
      integer head_lines, nx(lv), nw(lv)
      real*4 elonq(lv), elatq(lv), dx(lv), dw(lv)
      real*4 strq(lv), dipq(lv), rakeq(lv), dtop(lv), shyp(lv), dhyp(lv)
      real*4 astop(lv), sddp(lv,nq,np), rist(lv,nq,np), rupt(lv,nq,np)
      real*4 farea_in, zhyp, zhyp_max, pu
      real*4 stlon, stlat
      character*12 cap
      integer nlskip
c     /vmod_in/ layout, matching the main program exactly (:147).
      real*4 depth0(nlaymax), thic0(nlaymax), qp0(nlaymax), qs0(nlaymax)
      real*8 vp0(nlaymax), vsh0(nlaymax), rho0(nlaymax), vsmoho

      pu = 3.1415926/180

      read(5,'(a256)') slip_model
      read(5,'(a256)') velfile
      read(5,'(a256)') asite
      read(5,*) nsite
      read(5,'(a256)') outfile

      open(20, file=outfile(1:index(outfile,' ')-1),
     &     form='unformatted', access='stream', status='replace')

c---- slip model, verbatim from :251-285 --------------------------------
      open(10,file=slip_model(1:index(slip_model,' ')-1))
      read(10,*) nevnt
      zhyp_max = 0.0
      nstot = 0
      farea_in = 0.0
      do 133 iv=1,nevnt
         read(10,*) elonq(iv),elatq(iv),nx(iv),nw(iv),dx(iv),dw(iv)
         read(10,*) strq(iv),dipq(iv),rakeq(iv),dtop(iv),shyp(iv),dhyp(iv)
         nstot = nx(iv)*nw(iv) + nstot
         farea_in = nx(iv)*dx(iv)*nw(iv)*dw(iv) + farea_in
         astop(iv) = 0.5*nx(iv)*dx(iv)
         zhyp = dtop(iv) + dhyp(iv)/sin(dipq(iv)*pu)
         if(zhyp.gt.zhyp_max) zhyp_max = zhyp
         do j=1,nw(iv)
            read(10,*) (sddp(iv,i,j),i=1,nx(iv))
         end do
         do j=1,nw(iv)
            read(10,*) (rist(iv,i,j),i=1,nx(iv))
         end do
         do j=1,nw(iv)
            read(10,*) (rupt(iv,i,j),i=1,nx(iv))
         end do
133   continue
      close(10)

      write(20) nevnt, nstot
      write(20) farea_in, zhyp_max
      do iv = 1, nevnt
         write(20) nx(iv), nw(iv)
         write(20) elonq(iv), elatq(iv), dx(iv), dw(iv)
         write(20) strq(iv), dipq(iv), rakeq(iv), dtop(iv), shyp(iv),
     &             dhyp(iv), astop(iv)
         write(20) ((sddp(iv,i,j), i=1,nx(iv)), j=1,nw(iv))
         write(20) ((rist(iv,i,j), i=1,nx(iv)), j=1,nw(iv))
         write(20) ((rupt(iv,i,j), i=1,nx(iv)), j=1,nw(iv))
      enddo

c---- velocity model, verbatim from :318-349 and :494-513 ---------------
      vsmoho = -1.0
      vsmoho = 999.9
      open(10,file=velfile(1:index(velfile,' ')-1))
      read(10,*) j0
      jmoho = j0
      do 558 i=1,j0
         read(10,*) thic0(i),vp0(i),vsh0(i),rho0(i),qp0(i),qs0(i)
         depth0(i) = thic0(i)
         if(i.gt.1) depth0(i) = depth0(i) + depth0(i-1)
         if(vsh0(i).ge.vsmoho) then
            jmoho = i
            thic0(i) = 0.0
            depth0(i) = depth0(i-1)
            go to 559
         endif
558   continue
559   continue
      close(10)
      j0 = jmoho
      thic0(j0) = 0.0

      nlskip = -99
      if(depth0(1).gt.0.001.and.vp0(1).gt.0.01) then
         j0 = j0 + 1
         nlskip = nlskip + 1
         do 4433 i=j0,2,-1
            depth0(i) = depth0(i-1)
            thic0(i) = thic0(i-1)
            vp0(i) = vp0(i-1)
            vsh0(i) = vsh0(i-1)
            rho0(i) = rho0(i-1)
            qp0(i) = qp0(i-1)
            qs0(i) = qs0(i-1)
4433     continue
         depth0(1) = 0.0001
         thic0(1) = 0.0001
         vp0(1) = 0.001
         vsh0(1) = 0.0005
         rho0(1) = 0.001
      endif

      write(20) j0, nlskip
      write(20) (depth0(i), i=1,j0)
      write(20) (thic0(i), i=1,j0)
      write(20) (vp0(i), i=1,j0)
      write(20) (vsh0(i), i=1,j0)
      write(20) (rho0(i), i=1,j0)
      write(20) (qp0(i), i=1,j0)
      write(20) (qs0(i), i=1,j0)

c---- station list, verbatim from :822-857 ------------------------------
      open(1,file=asite(1:index(asite,' ')-1))
      head_lines = 0
      do
         read(1,*) dummy
         if ((dummy(1:1) .ne. '#') .AND. (dummy(1:1) .ne. '%')) exit
         head_lines=head_lines+1
      enddo
      rewind(1)
      do j=1,head_lines
         read(1,*)
      enddo
      write(20) head_lines
      do msite = 1, nsite
         read(1,*,end=1) stlon,stlat,cap
         write(20) stlon, stlat
         write(20) cap
      enddo
1     continue
      close(1)

      close(20)
      end
