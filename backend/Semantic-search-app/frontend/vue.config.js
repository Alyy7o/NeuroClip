const { defineConfig } = require('@vue/cli-service')
module.exports = {
  publicPath: '/',
  devServer: {
    allowedHosts: "all",
    host: '0.0.0.0',
    port: 8080,
    client: {
      webSocketURL: 'ws://localhost:8030/ws'
    }
  }
};