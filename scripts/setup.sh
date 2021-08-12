# Setup documents

# Generate a handout
cp ./thesis/slides.tex ./thesis/handout.tex
sed -i "1s/.*/\\\\documentclass\[handout\]\{beamer\}/" ./thesis/handout.tex
