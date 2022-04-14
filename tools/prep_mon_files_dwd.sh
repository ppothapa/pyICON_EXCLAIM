#!/usr/bin/bash

#set -x

module load cdo

expname=sml1 # 4 letters/digits
refdate=1870-01-01
refhour=00:00:00

cfu=~/workdir/ICON-Seamless/src/cfu/bin/cfu
itype_calendar=0

echo 'Calculate global mean of monthly mean for experiment' ${expname}':'

for file in `ls ${expname}_atm_2d_ml_????????????????.nc`
do
  sec1=`echo ${file} |cut -c1-9`
  sec2=mon
  sec3=`echo ${file} |cut -c15-34`
  date=`echo ${file} |cut -c16-23`
  date=`echo ${date}00`
  date_to_remove=`${cfu} get_next_dates ${date} 01:00:00 ${itype_calendar}`
  month_to_remove=`echo ${date_to_remove} | cut -c5-6`
  outfile=${sec1}${sec2}${sec3}
  echo ${outfile}
  cdo fldmean ${file} tmp1_${file}
  cdo setreftime,${refdate},${refhour} tmp1_${file} tmp2_${file}
  cdo expr,'radtot=sod_t-sou_t+thb_t;' tmp2_${file} tmp3_${file}
  cdo merge tmp2_${file} tmp3_${file} tmp4_${file}
  ncatted -a long_name,radtot,a,c,'total (sw_down-sw_up+lw_up) radiation at TOA' tmp4_${file} -o tmp5_${file}
  ncatted -a units,radtot,a,c,'W m-2' tmp5_${file} -o ${outfile}
  rm tmp*_${file}
done
