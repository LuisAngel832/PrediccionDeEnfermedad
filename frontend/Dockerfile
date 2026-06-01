# Imagen del FRONTEND: Nginx sirviendo el HTML/CSS/JS y haciendo proxy al backend.
FROM nginx:alpine

# Configuracion de Nginx (sirve estaticos + proxy /api -> backend)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Archivos del sitio
COPY frontend/index.html /usr/share/nginx/html/index.html
COPY frontend/style.css  /usr/share/nginx/html/style.css
COPY frontend/app.js     /usr/share/nginx/html/app.js

EXPOSE 80
