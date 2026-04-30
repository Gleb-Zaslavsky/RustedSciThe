*
*  -- Reference BLAS level2 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA
      INTEGER INCX,INCY,LDA,M,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION ZERO
      parameter(zero=0.0d+0)
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,IX,J,JY,KX
*     ..
*     .. External Subroutines ..
      EXTERNAL xerbla
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC max
*     ..
*
*     Test the input parameters.
*
      info = 0
      IF (m.LT.0) THEN
          info = 1
      ELSE IF (n.LT.0) THEN
          info = 2
      ELSE IF (incx.EQ.0) THEN
          info = 5
      ELSE IF (incy.EQ.0) THEN
          info = 7
      ELSE IF (lda.LT.max(1,m)) THEN
          info = 9
      END IF
      IF (info.NE.0) THEN
          CALL xerbla('DGER  ',info)
          RETURN
      END IF
*
*     Quick return if possible.
*
      IF ((m.EQ.0) .OR. (n.EQ.0) .OR. (alpha.EQ.zero)) RETURN
*
*     Start the operations. In this version the elements of A are
*     accessed sequentially with one pass through A.
*
      IF (incy.GT.0) THEN
          jy = 1
      ELSE
          jy = 1 - (n-1)*incy
      END IF
      IF (incx.EQ.1) THEN
          DO 20 j = 1,n
              IF (y(jy).NE.zero) THEN
                  temp = alpha*y(jy)
                  DO 10 i = 1,m
                      a(i,j) = a(i,j) + x(i)*temp
   10             CONTINUE
              END IF
              jy = jy + incy
   20     CONTINUE
      ELSE
          IF (incx.GT.0) THEN
              kx = 1
          ELSE
              kx = 1 - (m-1)*incx
          END IF
          DO 40 j = 1,n
              IF (y(jy).NE.zero) THEN
                  temp = alpha*y(jy)
                  ix = kx
                  DO 30 i = 1,m
                      a(i,j) = a(i,j) + x(ix)*temp
                      ix = ix + incx
   30             CONTINUE
              END IF
              jy = jy + incy
   40     CONTINUE
      END IF
*
      RETURN
*
*     End of DGER
*