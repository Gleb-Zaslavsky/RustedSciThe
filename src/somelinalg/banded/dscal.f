*  -- Reference BLAS level1 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION DA
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,M,MP1,NINCX
*     .. Parameters ..
      DOUBLE PRECISION ONE
      parameter(one=1.0d+0)
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC mod
*     ..
      IF (n.LE.0 .OR. incx.LE.0 .OR. da.EQ.one) RETURN
      IF (incx.EQ.1) THEN
*
*        code for increment equal to 1
*
*
*        clean-up loop
*
         m = mod(n,5)
         IF (m.NE.0) THEN
            DO i = 1,m
               dx(i) = da*dx(i)
            END DO
            IF (n.LT.5) RETURN
         END IF
         mp1 = m + 1
         DO i = mp1,n,5
            dx(i) = da*dx(i)
            dx(i+1) = da*dx(i+1)
            dx(i+2) = da*dx(i+2)
            dx(i+3) = da*dx(i+3)
            dx(i+4) = da*dx(i+4)
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         nincx = n*incx
         DO i = 1,nincx,incx
            dx(i) = da*dx(i)
         END DO
      END IF
      RETURN
*
*     End of DSCAL