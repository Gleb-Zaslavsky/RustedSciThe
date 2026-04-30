*> \brief \b DTRSM
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION ALPHA
*       INTEGER LDA,LDB,M,N
*       CHARACTER DIAG,SIDE,TRANSA,UPLO
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION A(LDA,*),B(LDB,*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DTRSM  solves one of the matrix equations
*>
*>    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
*>
*> where alpha is a scalar, X and B are m by n matrices, A is a unit, or
*> non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
*>
*>    op( A ) = A   or   op( A ) = A**T.
*>
*> The matrix X is overwritten on B.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] SIDE
*> \verbatim
*>          SIDE is CHARACTER*1
*>           On entry, SIDE specifies whether op( A ) appears on the left
*>           or right of X as follows:
*>
*>              SIDE = 'L' or 'l'   op( A )*X = alpha*B.
*>
*>              SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
*> \endverbatim
*>
*> \param[in] UPLO
*> \verbatim
*>          UPLO is CHARACTER*1
*>           On entry, UPLO specifies whether the matrix A is an upper or
*>           lower triangular matrix as follows:
*>
*>              UPLO = 'U' or 'u'   A is an upper triangular matrix.
*>
*>              UPLO = 'L' or 'l'   A is a lower triangular matrix.
*> \endverbatim
*>
*> \param[in] TRANSA
*> \verbatim
*>          TRANSA is CHARACTER*1
*>           On entry, TRANSA specifies the form of op( A ) to be used in
*>           the matrix multiplication as follows:
*>
*>              TRANSA = 'N' or 'n'   op( A ) = A.
*>
*>              TRANSA = 'T' or 't'   op( A ) = A**T.
*>
*>              TRANSA = 'C' or 'c'   op( A ) = A**T.
*> \endverbatim
*>
*> \param[in] DIAG
*> \verbatim
*>          DIAG is CHARACTER*1
*>           On entry, DIAG specifies whether or not A is unit triangular
*>           as follows:
*>
*>              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
*>
*>              DIAG = 'N' or 'n'   A is not assumed to be unit
*>                                  triangular.
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>           On entry, M specifies the number of rows of B. M must be at
*>           least zero.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>           On entry, N specifies the number of columns of B.  N must be
*>           at least zero.
*> \endverbatim
*>
*> \param[in] ALPHA
*> \verbatim
*>          ALPHA is DOUBLE PRECISION.
*>           On entry,  ALPHA specifies the scalar  alpha. When  alpha is
*>           zero then  A is not referenced and  B need not be set before
*>           entry.
*> \endverbatim
*>
*> \param[in] A
*> \verbatim
*>          A is DOUBLE PRECISION array, dimension ( LDA, k ),
*>           where k is m when SIDE = 'L' or 'l'
*>             and k is n when SIDE = 'R' or 'r'.
*>           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
*>           upper triangular part of the array  A must contain the upper
*>           triangular matrix  and the strictly lower triangular part of
*>           A is not referenced.
*>           Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
*>           lower triangular part of the array  A must contain the lower
*>           triangular matrix  and the strictly upper triangular part of
*>           A is not referenced.
*>           Note that when  DIAG = 'U' or 'u',  the diagonal elements of
*>           A  are not referenced either,  but are assumed to be  unity.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>           On entry, LDA specifies the first dimension of A as declared
*>           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
*>           LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
*>           then LDA must be at least max( 1, n ).
*> \endverbatim
*>
*> \param[in,out] B
*> \verbatim
*>          B is DOUBLE PRECISION array, dimension ( LDB, N )
*>           Before entry,  the leading  m by n part of the array  B must
*>           contain  the  right-hand  side  matrix  B,  and  on exit  is
*>           overwritten by the solution matrix  X.
*> \endverbatim
*>
*> \param[in] LDB
*> \verbatim
*>          LDB is INTEGER
*>           On entry, LDB specifies the first dimension of B as declared
*>           in  the  calling  (sub)  program.   LDB  must  be  at  least
*>           max( 1, m ).
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \ingroup trsm
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  Level 3 Blas routine.
*>
*>
*>  -- Written on 8-February-1989.
*>     Jack Dongarra, Argonne National Laboratory.
*>     Iain Duff, AERE Harwell.
*>     Jeremy Du Croz, Numerical Algorithms Group Ltd.
*>     Sven Hammarling, Numerical Algorithms Group Ltd.
*> \endverbatim
*>
*  =====================================================================

      SUBROUTINE dtrsm(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
*
*  -- Reference BLAS level3 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA
      INTEGER LDA,LDB,M,N
      CHARACTER DIAG,SIDE,TRANSA,UPLO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*)
*     ..
*
*  =====================================================================
*
*     .. External Functions ..
      LOGICAL LSAME
      EXTERNAL lsame
*     ..
*     .. External Subroutines ..
      EXTERNAL xerbla
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC max
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,J,K,NROWA
      LOGICAL LSIDE,NOUNIT,UPPER
*     ..
*     .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      parameter(one=1.0d+0,zero=0.0d+0)
*     ..
*
*     Test the input parameters.
*
      lside = lsame(side,'L')
      IF (lside) THEN
          nrowa = m
      ELSE
          nrowa = n
      END IF
      nounit = lsame(diag,'N')
      upper = lsame(uplo,'U')
*
      info = 0
      IF ((.NOT.lside) .AND. (.NOT.lsame(side,'R'))) THEN
          info = 1
      ELSE IF ((.NOT.upper) .AND. (.NOT.lsame(uplo,'L'))) THEN
          info = 2
      ELSE IF ((.NOT.lsame(transa,'N')) .AND.
     +         (.NOT.lsame(transa,'T')) .AND.
     +         (.NOT.lsame(transa,'C'))) THEN
          info = 3
      ELSE IF ((.NOT.lsame(diag,'U')) .AND.
     +         (.NOT.lsame(diag,'N'))) THEN
          info = 4
      ELSE IF (m.LT.0) THEN
          info = 5
      ELSE IF (n.LT.0) THEN
          info = 6
      ELSE IF (lda.LT.max(1,nrowa)) THEN
          info = 9
      ELSE IF (ldb.LT.max(1,m)) THEN
          info = 11
      END IF
      IF (info.NE.0) THEN
          CALL xerbla('DTRSM ',info)
          RETURN
      END IF
*
*     Quick return if possible.
*
      IF (m.EQ.0 .OR. n.EQ.0) RETURN
*
*     And when  alpha.eq.zero.
*
      IF (alpha.EQ.zero) THEN
          DO 20 j = 1,n
              DO 10 i = 1,m
                  b(i,j) = zero
   10         CONTINUE
   20     CONTINUE
          RETURN
      END IF
*
*     Start the operations.
*
      IF (lside) THEN
          IF (lsame(transa,'N')) THEN
*
*           Form  B := alpha*inv( A )*B.
*
              IF (upper) THEN
                  DO 60 j = 1,n
                      IF (alpha.NE.one) THEN
                          DO 30 i = 1,m
                              b(i,j) = alpha*b(i,j)
   30                     CONTINUE
                      END IF
                      DO 50 k = m,1,-1
                          IF (b(k,j).NE.zero) THEN
                              IF (nounit) b(k,j) = b(k,j)/a(k,k)
                              DO 40 i = 1,k - 1
                                  b(i,j) = b(i,j) - b(k,j)*a(i,k)
   40                         CONTINUE
                          END IF
   50                 CONTINUE
   60             CONTINUE
              ELSE
                  DO 100 j = 1,n
                      IF (alpha.NE.one) THEN
                          DO 70 i = 1,m
                              b(i,j) = alpha*b(i,j)
   70                     CONTINUE
                      END IF
                      DO 90 k = 1,m
                          IF (b(k,j).NE.zero) THEN
                              IF (nounit) b(k,j) = b(k,j)/a(k,k)
                              DO 80 i = k + 1,m
                                  b(i,j) = b(i,j) - b(k,j)*a(i,k)
   80                         CONTINUE
                          END IF
   90                 CONTINUE
  100             CONTINUE
              END IF
          ELSE
*
*           Form  B := alpha*inv( A**T )*B.
*
              IF (upper) THEN
                  DO 130 j = 1,n
                      DO 120 i = 1,m
                          temp = alpha*b(i,j)
                          DO 110 k = 1,i - 1
                              temp = temp - a(k,i)*b(k,j)
  110                     CONTINUE
                          IF (nounit) temp = temp/a(i,i)
                          b(i,j) = temp
  120                 CONTINUE
  130             CONTINUE
              ELSE
                  DO 160 j = 1,n
                      DO 150 i = m,1,-1
                          temp = alpha*b(i,j)
                          DO 140 k = i + 1,m
                              temp = temp - a(k,i)*b(k,j)
  140                     CONTINUE
                          IF (nounit) temp = temp/a(i,i)
                          b(i,j) = temp
  150                 CONTINUE
  160             CONTINUE
              END IF
          END IF
      ELSE
          IF (lsame(transa,'N')) THEN
*
*           Form  B := alpha*B*inv( A ).
*
              IF (upper) THEN
                  DO 210 j = 1,n
                      IF (alpha.NE.one) THEN
                          DO 170 i = 1,m
                              b(i,j) = alpha*b(i,j)
  170                     CONTINUE
                      END IF
                      DO 190 k = 1,j - 1
                          IF (a(k,j).NE.zero) THEN
                              DO 180 i = 1,m
                                  b(i,j) = b(i,j) - a(k,j)*b(i,k)
  180                         CONTINUE
                          END IF
  190                 CONTINUE
                      IF (nounit) THEN
                          temp = one/a(j,j)
                          DO 200 i = 1,m
                              b(i,j) = temp*b(i,j)
  200                     CONTINUE
                      END IF
  210             CONTINUE
              ELSE
                  DO 260 j = n,1,-1
                      IF (alpha.NE.one) THEN
                          DO 220 i = 1,m
                              b(i,j) = alpha*b(i,j)
  220                     CONTINUE
                      END IF
                      DO 240 k = j + 1,n
                          IF (a(k,j).NE.zero) THEN
                              DO 230 i = 1,m
                                  b(i,j) = b(i,j) - a(k,j)*b(i,k)
  230                         CONTINUE
                          END IF
  240                 CONTINUE
                      IF (nounit) THEN
                          temp = one/a(j,j)
                          DO 250 i = 1,m
                              b(i,j) = temp*b(i,j)
  250                     CONTINUE
                      END IF
  260             CONTINUE
              END IF
          ELSE
*
*           Form  B := alpha*B*inv( A**T ).
*
              IF (upper) THEN
                  DO 310 k = n,1,-1
                      IF (nounit) THEN
                          temp = one/a(k,k)
                          DO 270 i = 1,m
                              b(i,k) = temp*b(i,k)
  270                     CONTINUE
                      END IF
                      DO 290 j = 1,k - 1
                          IF (a(j,k).NE.zero) THEN
                              temp = a(j,k)
                              DO 280 i = 1,m
                                  b(i,j) = b(i,j) - temp*b(i,k)
  280                         CONTINUE
                          END IF
  290                 CONTINUE
                      IF (alpha.NE.one) THEN
                          DO 300 i = 1,m
                              b(i,k) = alpha*b(i,k)
  300                     CONTINUE
                      END IF
  310             CONTINUE
              ELSE
                  DO 360 k = 1,n
                      IF (nounit) THEN
                          temp = one/a(k,k)
                          DO 320 i = 1,m
                              b(i,k) = temp*b(i,k)
  320                     CONTINUE
                      END IF
                      DO 340 j = k + 1,n
                          IF (a(j,k).NE.zero) THEN
                              temp = a(j,k)
                              DO 330 i = 1,m
                                  b(i,j) = b(i,j) - temp*b(i,k)
  330                         CONTINUE
                          END IF
  340                 CONTINUE
                      IF (alpha.NE.one) THEN
                          DO 350 i = 1,m
                              b(i,k) = alpha*b(i,k)
  350                     CONTINUE
                      END IF
  360             CONTINUE
              END IF
          END IF
      END IF
*
      RETURN
*
*     End of DTRSM
*

      END