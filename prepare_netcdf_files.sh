#!/bin/bash

# xarray cannot read REMO's netcdf files directly because of https://github.com/pydata/xarray/issues/2368
# To fix this, I have to rename any variables that have the same name as an attribute in the variable.
# This script also remaps EUR-44 data to the E-OBS grid.
# Also, this aggregates hourly data to days.

# REQUIREMENTS: nco (or more specifically: ncrename) and cdo (for regridding)
# On MacOS: brew install nco cdo

display_usage() { 
	echo -e "\nUsage:\n./prepare_netcdf_files.sh path [-r] [-m] \n"
    echo -e "This prepares .nc files recursively in path"
    echo -e "-d: Aggregate daily"
    echo -e "-f: Filter data points for year"
    echo -e "-k: Keep intermediate files"
    echo -e "-m: Aggregate monthly"
    echo -e "-r: Only rename variables (no interpolation etc.)"
    echo -e "-s: Aggregate with sum instead of mean"
}

# if less than two arguments supplied, display usage 
if [  $# -le 0 ] 
then 
    display_usage
    exit 1
fi

rename_only_flag=''
daily_flag=''
filter_year_flag=''
keep_flag=''
monthly_flag=''
sum_flag=''

while getopts 'dfkmrs' flag ${@:2}; do
  case "${flag}" in
    d) daily_flag='true' ;;
    f) filter_year_flag='true' ;;
    k) keep_flag='true' ;;
    m) monthly_flag='true' ;;
    r) rename_only_flag='true' ;;
    s) sum_flag='true' ;;
    *) display_usage
       exit 1 ;;
  esac
done

PATH=$1

# Write result of find to bash array, see: https://stackoverflow.com/a/23357277
files=()
while IFS=  read -r -d $'\0'; do
    files+=("$REPLY")
# Find all *.nc files but not ones with *_remap.nc, *_monmean.nc, or *_crop.nc
done < <(/usr/bin/find $PATH -type f -name "*.nc" ! -name '*_remap.nc' ! -name '*_monagg.nc' ! -name '*_crop.nc' -print0)

# sort files
IFS=$'\n' files=($(/usr/bin/sort <<<"${files[*]}"))
unset IFS

if [ "$rename_only_flag" != 'true' ] && [ "$monthly_flag" == 'true' ]; then
    echo "Concatenating files..."
    /usr/local/bin/cdo cat ${files[@]} ${files%/*}/merged.nc
    files=("${files%/*}/merged.nc")
fi

index=1
for file in ${files[@]}; do

    if [[ ! -e "$file" ]]; then
        ((index++))
        continue
    fi

    filename="${file%.*}"

    if [ "$rename_only_flag" != 'true' ]; then
        echo "Processing $filename ($index/${#files[@]})"

        input="$file"
        output="${filename}_crop.nc"
        /usr/local/bin/cdo sellonlatbox,-1.426746,22.217735,42.769917,57.060219 "$input" "$output"
        input=$output

        if [ "$filter_year_flag" == 'true' ]; then
            output="${filename}_filtered.nc"

            [[ $file =~ (?!_)[0-9]{4}(?=[0-9]{2}?.nc) ]]
            cur_year=$(echo $file | /usr/local/bin/ggrep -oP '(?!_)[0-9]{4}(?=[0-9]{2}?.nc)')
            echo "$cur_year"
            /usr/local/bin/cdo selyear,"$cur_year" "$input" "$output"

            if [ "$keep_flag" != 'true' ]; then
                /bin/rm "$input"
            fi
            input=$output
        fi

        if [ "$monthly_flag" == 'true' ]; then
            echo "Aggregate monthly..."
            output="${filename}_monagg.nc"
            if [ "$sum_flag" == 'true' ]; then
                echo "Aggregation method: sum"
                /usr/local/bin/cdo monsum "$input" "$output"
            else
                echo "Aggregation method: mean"
                /usr/local/bin/cdo monmean "$input" "$output"
            fi

            if [ "$keep_flag" != 'true' ]; then
                /bin/rm "$input"
            fi

            input=$output
        elif [ "$daily_flag" == 'true' ]; then
            echo "Aggregate daily..."
            output="${filename}_dayagg.nc"
            if [ "$sum_flag" == 'true' ]; then
                echo "Aggregation method: sum"
                /usr/local/bin/cdo daysum "$input" "$output"
            else
                echo "Aggregation method: mean"
                /usr/local/bin/cdo daymean "$input" "$output"
            fi

            if [ "$keep_flag" != 'true' ]; then
                /bin/rm "$input"
            fi

            input=$output
        fi

        output="${filename}_remap.nc"
        # Remapping REMO to EOBS with bicubic interpolation like done in the DeepSD paper as a first step
        # COMMENT THE NEXT LINE IF YOU APPLY IT TO EOBS!
        # The two fillmiss2 commands are for GER-11 as the last cell row and the last column are missing for
        # some reason. The second fillmiss2 fills the last remaining pixel all the way on the corner.
        /usr/local/bin/cdo -remapbic,data/DEM_GER-11_GTOPO_remap.nc -fillmiss2 -fillmiss2 "$input" "$output"

        if [ "$keep_flag" != 'true' ]; then
            /bin/rm "$input"
        fi
    fi

    /usr/local/bin/ncrename -v time,time_var -v lon,lon_var -v lat,lat_var "$output"
    ((index++))
done


# cdo -remapbic,../../GTOPO/DEM_GER-11_GTOPO_remap.nc e031001e_c260_200001.nc e031001e_c260_200001_remapbic.nc