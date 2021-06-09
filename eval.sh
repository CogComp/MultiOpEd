
echo "BART baseline" >> results/results.txt

for seed in 1 6 9
do
python eval.py --dataset_path data/MultiOpEd.csv --generated_file outputs/bart_baseline/seed=$seed/test_generated.txt --labels_file outputs/bart_baseline/seed=$seed/test_labels.txt --relevance_classifier_path /path/to/relevance/classifier --stance_classifier_path /path/to/stance/classifier --result_path results
done

echo "BART+Relevance" >> results/results.txt

for seed in 1 6 9
do
python eval.py --dataset_path data/MultiOpEd.csv --generated_file outputs/bart+relevance/seed=$seed/test_generated.txt --labels_file outputs/bart+relevance/seed=$seed/test_labels.txt --relevance_classifier_path /path/to/relevance/classifier --stance_classifier_path /path/to/stance/classifier --result_path results
done

echo "BART+stance" >> results/results.txt

for seed in 1 6 9
do
python eval.py --dataset_path data/MultiOpEd.csv --generated_file outputs/bart+stance/seed=$seed/test_generated.txt --labels_file outputs/bart+stance/seed=$seed/test_labels.txt --relevance_classifier_path /path/to/relevance/classifier --stance_classifier_path /path/to/stance/classifier --result_path results
done


echo "BART+both" >> results/results.txt
for seed in 1 6 9
do
python eval.py --dataset_path data/MultiOpEd.csv --generated_file outputs/bart+both/seed=$seed/test_generated.txt --labels_file outputs/bart+both/seed=$seed/test_labels.txt --relevance_classifier_path /path/to/relevance/classifier --stance_classifier_path /path/to/stance/classifier --result_path results
done

