// src/services/api.js
import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api'
})

export default {
  getTopics() {
    return apiClient.get('/topics')
  },
  getMessages(topic, limit) {
    return apiClient.get('/messages', { params: { topic, limit } })
  },
  initDefaultMessages() {
    return apiClient.post('/messages/init')
  },
  sendCustomMessages(topic, payload) {
    return apiClient.post('/messages/add', payload, { params: { topic } })
  }
}