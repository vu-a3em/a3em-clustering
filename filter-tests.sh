OUTPUT=zzz-filter-test-results
CMD="time python sim2.py --events /home/devin/Downloads/dog-events/ /home/devin/Downloads/misc-events/ --backgrounds /home/devin/Downloads/edansa-events/ --event-prob 0.75"

rm -rf $OUTPUT
mkdir $OUTPUT

$CMD --event-freqs Dog:1 Rooster:1 Aircraft:1 >$OUTPUT/1.txt 2>&1 &
$CMD --event-freqs Dog:4 Rooster:1 Aircraft:1 >$OUTPUT/2.txt 2>&1 &
$CMD --event-freqs Dog:16 Rooster:1 Aircraft:1 >$OUTPUT/3.txt 2>&1 &
$CMD --event-freqs Dog:64 Rooster:1 Aircraft:1 >$OUTPUT/4.txt 2>&1 &
wait
