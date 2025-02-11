# backend/Dockerfile

FROM python:3.10.8

RUN mkdir -p app/backend

COPY Backend/requirements.txt app/backend/requirements.txt
COPY Data app/Data

COPY Backend app/backend

WORKDIR /app/backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080","--reload"]
