mkdir reference_tmp
find ./reference -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) -exec cp {} ./reference_tmp \;
find ./reference_3W -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) -exec cp {} ./reference_3W_total \;
mkdir prediction_tmp
find ./prediction -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) -exec cp {} ./prediction_tmp \;
find ./prediction_top4096_maskTrue_v3 -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) -exec cp {} ./prediction_top4096_maskTrue_v3_total \;