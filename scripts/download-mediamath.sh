path_start=$(pwd)
base_name=$(basename $path_start)

path_local_ctr=../data/raw/mm-ctr/
path_local_cpc=../data/raw/mm-cpc/

path_aws_bucket=""
path_aws_ctr=""
path_aws_cpc=""

function download_data {
    if [ -d $2 ]; then
        echo "MediaMath $3 dataset already exists";
    else
        echo "Downloading MediaMath $3 dataset";
        mkdir -p $2
        cd $2

        aws s3 cp --recursive --no-sign-request $1 .

        mv train/data_class0.csv train/data-negative
        mv train/data_class1.csv train/data-positive

        mv test/data.csv/* test/
        rmdir test/data.csv/

        mv validation/data.csv/* validation/
        rmdir validation/data.csv/

        cd $path_start
    fi;
}

if [ $base_name = 'scripts' ]; then
    download_data $path_aws_ctr $path_local_ctr CTR
    download_data $path_aws_cpc $path_local_cpc CPC 
else
    echo "Please cd into scripts directory first."
fi
