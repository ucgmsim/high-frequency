c=======================================================================
c rng_driver.f -- emit golden output for the RNG kernels.
c
c Links against reference/pcg32.f and reference/hb_high_subs.f so the
c goldens come from the same code the oracle binary runs.
c
c stdin:  mode
c         seed
c         count
c         output-file
c
c mode 1  raw pcg32_next() stream        -> integer*8 per draw
c mode 2  rand_numb(0) stream            -> real*4 per draw
c mode 3  normal_random_number(count,.)  -> real*4 per element
c mode 4  RANU2(count,.)                 -> real*4 per element
c
c All output is access='stream' (no record markers) little-endian on x86-64.
c The 32-bit generator output is widened to integer*8 so the file format is
c unambiguous about sign.
c
c mode 0 prints the mutated seed to stdout instead: init_random_seed
c increments its argument, and hb_high reads that mutated value at line 1366.
c=======================================================================
      program rng_driver

      implicit none

      integer mode, seed, count, i, seed_in
      character*256 outfile
      integer*8 pcg32_next
      real*4 rand_numb
      real*4, allocatable :: buf(:)

      read(5,*) mode
      read(5,*) seed
      read(5,*) count
      read(5,'(a256)') outfile

      seed_in = seed
      call init_random_seed(seed)

      if (mode .eq. 0) then
c        Report the argument mutation: seed_out - seed_in must equal the
c        seed-word count pinned in pcg32.f.
         write(6,'(i0,1x,i0)') seed_in, seed
         stop
      endif

      open(10, file=outfile(1:index(outfile,' ')-1),
     &     form='unformatted', access='stream', status='replace')

      if (mode .eq. 1) then
         do i = 1, count
            write(10) pcg32_next()
         enddo

      else if (mode .eq. 2) then
         do i = 1, count
            write(10) rand_numb(0)
         enddo

      else if (mode .eq. 3) then
         allocate(buf(count))
         call normal_random_number(count, buf)
         write(10) (buf(i), i = 1, count)
         deallocate(buf)

      else if (mode .eq. 4) then
         allocate(buf(count))
         call RANU2(count, buf)
         write(10) (buf(i), i = 1, count)
         deallocate(buf)

      else
         write(0,*) 'unknown mode ', mode
         close(10)
         stop 1
      endif

      close(10)

      end
