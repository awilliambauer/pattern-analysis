#!/homes/gws/awb/.local/bin/zsh
while read a b
do
    if [[ ! -f tmp_data/$a.atoms || ! -f tmp_data/$b.atoms ]]; then
        # if atoms files are missing, result is nan
        echo "$a $b nan"
    else
        timeout 120 zsh -c "./TMscore tmp_data/$a.atoms tmp_data/$b.atoms | sed -n -E -e 's/TM-score\s+= ([0-9]+\.[0-9]+).*/$a $b \1/p'"
        if [[ $? -eq 124 ]]; then
            # if it times out, result is nan
            echo "$a $b nan"
        fi
     fi
done < "$1"
