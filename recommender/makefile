test:
	docker compose -f docker-compose.yml up --build --force-recreate -d
	docker exec recommender python3 -m pip install -r requirements.test.txt
	docker exec recommender /home/work/wait-for-it.sh -t 30 recommender:8001 -- nose2 -v
	docker compose down -v

	
