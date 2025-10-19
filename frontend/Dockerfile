FROM node:20-alpine AS build

WORKDIR /app

RUN apk update && apk upgrade && rm -rf /var/cache/apk/*

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .

ARG VITE_BACKEND_URL=http://localhost:8000
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

RUN npm run build

FROM nginx:1.27-alpine

RUN apk update && apk upgrade && rm -rf /var/cache/apk/*

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

RUN chown -R nginx:nginx /usr/share/nginx/html /etc/nginx/conf.d && \
    mkdir -p /var/cache/nginx && \
    chown -R nginx:nginx /var/cache/nginx

EXPOSE 5423

USER nginx

CMD ["nginx", "-g", "daemon off;"]
