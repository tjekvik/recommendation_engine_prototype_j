# Use official Node.js LTS image
FROM node:20-alpine3.22

# Set working directory
WORKDIR /app

RUN apk add --update python3 make g++\
   && rm -rf /var/cache/apk/*

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy application source
COPY . .

# Expose port (change if your app uses a different port)
EXPOSE 4000