<template>
  <div id="app">
    <header class="app-header">
      <div class="logo">
        <div style="text-align: center;">
          <h1>TL;DЯ</h1>
          <p>Пересказ - это быстро</p>
        </div>
      </div>
    </header>

    <main class="app-content">
      <ChatViewer />
    </main>
  </div>
</template>

<script>
import ChatViewer from './views/ChatViewer.vue'

export default {
  components: {
    ChatViewer
  }
}
</script>

<style lang="scss">
.app-header {
  background-color: #0d0d0d;
  color: #4caf50;
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);

  .logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;

    h1 {
      font-size: 1.8rem;
      margin: 0;
      font-weight: bold;
    }

    .logo-icon {
      font-size: 1.8rem;
      color: #4caf50;
    }
  }
}
</style>