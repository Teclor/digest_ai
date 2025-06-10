<template>
  <div class="tabs-wrapper">
    <el-tabs v-model="activeTab" type="card" class="chat-tabs">
      <el-tab-pane label="Сообщения" name="messages">
        <MessageList
          :topics="topics"
          :selected-topic="selectedTopic"
          :limit="limit"
          :messages="messages"
          @update:selected-topic="selectedTopic = $event"
          @update:limit="limit = $event"
          @load-messages="loadMessages"
        />
      </el-tab-pane>

      <el-tab-pane label="Отправить JSON" name="send-json">
        <ChatForm @submit="sendCustomJson" @initDefaultMessages="initDefaultMessages"/>
      </el-tab-pane>

      <el-tab-pane label="JS-код" name="js-code">
        <JsCodeBlock :topic="selectedTopic" :limit="limit" @copied="onCopy" />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
import MessageList from '@/components/MessageList.vue'
import ChatForm from '@/components/ChatForm.vue'
import JsCodeBlock from '@/components/JsCodeBlock.vue'
import api from '@/services/api'

export default {
  components: {
    MessageList,
    ChatForm,
    JsCodeBlock
  },
  data() {
    return {
      activeTab: 'messages',
      topics: [],
      selectedTopic: null,
      limit: 10,
      messages: []
    }
  },
  async mounted() {
    try {
      const res = await api.getTopics()
      this.topics = Array.isArray(res.data) ? res.data : ['news', 'chat']
      if (this.topics.length > 0) this.selectedTopic = this.topics[0]
    } catch (err) {
      console.error('Ошибка при загрузке топиков', err)
      this.topics = ['news', 'chat']
      this.selectedTopic = this.topics[0]
    }
  },
  methods: {
    async loadMessages({ topic, limit }) {
      try {
        const res = await api.getMessages(topic, limit)
        this.messages = res.data.messages
      } catch (err) {
        alert('Ошибка при получении сообщений: ' + err.message)
      }
    },
    async sendCustomJson({ topic, type, json }) {
      const fullTopicName = `${type}_${topic}`

      let parsedJson
      try {
        parsedJson = JSON.parse(json)
      } catch (e) {
        alert("Некорректный JSON")
        return
      }

      const payload = {
        messages: Array.isArray(parsedJson) ? parsedJson : [parsedJson]
      }

      try {
        await api.sendCustomMessages(fullTopicName, payload)
        alert(`Сообщения успешно отправлены в топик ${fullTopicName}`)
      } catch (err) {
        alert("Ошибка при отправке сообщений: " + (err.response?.data?.detail || err.message))
      }
    },
    async initDefaultMessages() {
      if (!confirm("Вы уверены, что хотите заполнить топики данными из файлов?")) {
        return
      }

      try {
        api.initDefaultMessages()
        alert("Данные успешно заполнены!")
      } catch (err) {
        alert("Ошибка при инициализации данных: " + (err.response?.data?.detail || err.message))
      }
    },
    onCopy() {
      alert('JS-код скопирован в буфер обмена!')
    }
  }
}
</script>