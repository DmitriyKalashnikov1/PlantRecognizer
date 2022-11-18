FROM python:3.8
RUN mkdir -p /usr/src/plantrecognizer/
WORKDIR /usr/src/plantrecognizer/

COPY ./*.csv ./*.py ./*.txt /usr/src/plantrecognizer/

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", 'model_cli.py -i input.csv -o output.csv']
