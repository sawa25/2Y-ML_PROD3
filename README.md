все запущено на ubuntu, (start,second - вспомогательные файлы для проверки модели и установки библиотек- их можно не рассматривать)

Учебный проект, реализующий 2 техники:

1.модель принимает частями данные и с каждой новой порцией дообучается от предыдущего состояния.
Т.о. постепенно происходит корректировка весов с учетом новых поступающих данных.

2.процесс мониторится через prometheus->grafana

prometheus запущен через докер:
sudo docker run --network=host -v /etc/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
при этом без параметра --network=host не получалось пустить в прометеус данные от клиента http://localhost:8000/metrics
и доступен по http://localhost:9090

настройки в prometheus.yml:
scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
  - job_name: "my_model_metrics"
    static_configs:
      - targets: ["localhost:8000"]

графана запущена sudo systemctl start grafana-server
и доступна по http://localhost:3000

код питон prepare.py запущен в бесконечный цикл в VSCode.

при этом графана отображает в реальном времени значения трех метрик, которые обновляются в процессе дообучения модели.
более подробно - в prepare.py.
