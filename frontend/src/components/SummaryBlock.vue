<template>
  <!-- Блок показывается только если есть сообщения -->
  <div v-if="messages.length > 0" class="summary-block">
    <!-- Обрезанный текст -->
    <div v-if="truncatedText" class="text-block">
      <h3>Текст для пересказа</h3>
      <pre class="truncated-text">{{ truncatedText }}</pre>

      <!-- Кнопка внутри блока -->
      <button class="btn-summarize" @click="getSummary">Пересказать</button>
    </div>
    <div v-else>
      Подготовка текста...
    </div>

    <!-- Результат пересказа -->
    <div v-if="summary" class="summary-result">
      <h3>Результат пересказа</h3>
      <pre class="summary-text">{{ summary }}</pre>
    </div>
  </div>
</template>


<script>
import axios from 'axios'

export default {
  props: {
    messages: {
      type: Array,
      required: true
    },
    selectedTopic: {
      type: String,
      default: () => 'chat_1'
    }
  },
  data() {
    return {
      truncatedText: '',
      summary: '',
      isLoading: false
    }
  },
  watch: {
    messages: {
      deep: true,
      handler(newMessages) {
        if (newMessages.length > 0) {
          this.getTruncatedText()
        } else {
          this.truncatedText = ''
          this.summary = ''
        }
      }
    }
  },
  methods: {
    async getTruncatedText() {
      const text = this.messages.map(m => m.data.text).join('\n')

      try {
        const response = await axios.post('/api/truncated', text)
        this.truncatedText = response.data
      } catch (err) {
        console.error('Ошибка при обрезке текста:', err)
        this.truncatedText = ''
      }
    },

    async getSummary() {
      if (!this.truncatedText) {
        alert('Нет текста для пересказа')
        return
      }

      try {
        const response = await axios.post('/api/summary', {
          text: this.truncatedText,
          topic: this.selectedTopic
        })

        this.summary = response.data.summary
      } catch (err) {
        console.error('Ошибка при получении пересказа:', err)
        this.summary = 'Не удалось получить пересказ'
      }
    }
  }
}
</script>

<style scoped>
.summary-block {
  margin-top: 2rem;
  background-color: #1f1f1f;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.text-block {
  margin-bottom: 1.5rem;
}

pre {
  background-color: #1a1a1a;
  color: #4caf50;
  font-family: monospace;
  font-size: 0.95rem;
  padding: 1rem;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}

.btn-summarize {
  display: inline-block;
  margin-top: 1rem;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  color: white;
  background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-summarize:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
}

.summary-result {
  margin-top: 1.5rem;
}

.summary-text {
  background-color: #2a2a2a;
  color: #81c784;
}
</style>