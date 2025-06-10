<template>
  <div class="mb-8">
    <h3 class="text-xl font-bold mb-3 text-white">JS-код для получения сообщений чата</h3>
    <label class="block mb-2 text-gray-300">Скопируйте этот код:</label>

    <div class="relative">
      <textarea
        ref="jsCodeField"
        :value="jsCodeExample"
        readonly
        class="w-full h-32 p-4 bg-gray-900 text-green-400 rounded-lg focus:outline-none resize-none pr-28 font-mono text-sm leading-relaxed shadow-inner"
      ></textarea>

      <button
        @click="copyJsCode"
        class="absolute right-3 bottom-3 px-4 py-2 bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-700 hover:to-gray-600 text-white rounded-md transition-all duration-200 shadow-md hover:shadow-lg flex items-center gap-2"
      >
        📋 Скопировать
      </button>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    topic: String,
    limit: Number
  },
  computed: {
    jsCodeExample() {
      const topic = this.topic || 'chat_1'
      const limit = this.limit
      return `(() => console.log(JSON.stringify([...document.querySelectorAll('div.chat-message__text')].map(m => ({text: m.textContent.trim() || ''})))))()`
    }
  },
  methods: {
    copyJsCode() {
      const textarea = this.$refs.jsCodeField
      textarea.select()
      document.execCommand('copy')
      this.$emit('copied')
    }
  }
}
</script>