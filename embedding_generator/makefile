test:
	docker compose -f docker-compose.yml up --build --force-recreate -d
	docker exec embedding-generator python3 -m pip install -r requirements.test.txt
	docker exec embedding-generator /home/work/wait-for-it.sh -t 30 embedding-generator:8000 -- nose2 -v
	docker compose down -v

init_model:
	python -m pip install -r requirements.txt
	python utils/download_model.py
	
