<template>
  <div style="padding: 2rem; font-family: sans-serif">
    <h1>Kafka Chat Viewer</h1>

    <div v-if="topics.length === 0">
      <p>🔄 Загрузка топиков...</p>
    </div>

    <div v-else>
      <label for="topic">Выберите топик:</label>
      <select v-model="selectedTopic">
        <option v-for="topic in topics" :key="topic" :value="topic">{{ topic }}</option>
      </select>

      <label for="limit" style="margin-left: 1rem;">Количество сообщений:</label>
      <input type="number" v-model.number="limit" min="1" max="1000" />

      <button @click="loadMessages" style="margin-left: 1rem;">Загрузить</button>
    </div>

    <div v-if="messages.length > 0" style="margin-top: 2rem;">
      <h2>Последние сообщения:</h2>
      <div v-for="(msg, idx) in messages" :key="idx" style="border-bottom: 1px solid #ccc; padding: 0.5rem 0;">
        <pre>{{ msg }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      topics: [],
      selectedTopic: null,
      limit: 10,
      messages: []
    }
  },
  async mounted() {
    try {
      const res = await axios.get('/api/topics')
      this.topics = res.data
      if (this.topics.length > 0) {
        this.selectedTopic = this.topics[0]
      }
    } catch (err) {
      alert('Ошибка при загрузке топиков: ' + err)
    }
  },
  methods: {
    async loadMessages() {
      if (!this.selectedTopic) return
      try {
        const res = await axios.get('/api/messages', {
          params: {
            topic: this.selectedTopic,
            limit: this.limit
          }
        })
        this.messages = res.data.messages
      } catch (err) {
        alert('Ошибка при получении сообщений: ' + err)
      }
    }
  }
}
</script>
