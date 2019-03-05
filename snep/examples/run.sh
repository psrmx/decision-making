#!/bin/bash

PROJNAME=${2%.*}
ROOTDIR=$HOME/experiments/$PROJNAME
DATETIME=$( date "+%Y-%m-%d-%Hh%Mm%Ss" )
NUMPROCS=$(( $1 + 1 ))
JOBDIR=$ROOTDIR/$DATETIME

echo "Writing job output to $JOBDIR"
mkdir $JOBDIR

qsub 				\
-N $PROJNAME			\
-V				\
-q cognition-all.q		\
-pe cognition.pe $NUMPROCS	\
-cwd				\
-o $JOBDIR/stdout		\
-e $JOBDIR/stderr		\
-b y 				\
-v RT=$3 JOBDIR=$JOBDIR NUMPROCS=$NUMPROCS \
python $2

#-S /bin/bash
# -np $NSLOTS -np 2 
## $-o foo.o${JOB_ID/.*} 
## $-e foo.e${JOB_ID/.*} 
