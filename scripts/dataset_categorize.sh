# for f in *.jpg; do
#     dir=`echo $f | awk -F'[._]' '{ for(i = 1; i<NF-1; i++) printf "%s",$i OFS}' | tr ' ' '_'`
#     mkdir -p "$dir"
#     mv "$f" "$dir"
# done
# 

find . -type d -print0 | while read -d '' -r dir; do
    files=("$dir"/*)
    printf "%5d files in directory %s\n" "${#files[@]}" "$dir"
    if [ "$dir" = '.' ]; then
        echo error
    else
        train="train/"`echo $dir | tr -d ./`
        eval="eval/"`echo $dir | tr -d ./`
        echo $train $eval
        for ((i=1; i<=$((${#files[@]} * 9 / 10)); i++)); do
            mkdir -p "$train"
            mv "$dir"/`echo $dir | tr -d ./`"$i".jpg "$train"
        done
        mkdir -p "$eval"
        mv "$dir"/`echo $dir | tr -d ./`*.jpg "$eval"

    fi

done
