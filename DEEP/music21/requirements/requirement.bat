rem if run from terminal, (type file name or double click), conda ,not found and requirement_conda is empty
rem go in conda gui, enable tf2-gpu, create a powershell prompt,  and run from there

echo 'create conda and pip requirements list'

rem pip freeze includes file path

pip list --format=freeze > my_requirement_pip.txt
conda list --export > my_requirement_conda.txt

conda env export -n tf24 > condapip_requirement.yml

rem pip install -r requirements.txt
rem conda install --file requirements.txt
