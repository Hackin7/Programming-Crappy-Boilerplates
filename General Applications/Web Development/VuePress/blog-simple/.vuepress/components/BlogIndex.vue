<!-- /.vuepress/components/BlogIndex.vue -->

<template>
  <ol class="post-container">
    <router-link v-for="page in pages" :to="page.path">
      <li>
        <div class="page-detail">
          <div class="page-title">{{ page.title }}</div>
          <div class="page-description">{{ page.frontmatter.description }}</div>
          <div class="page-author">Author: {{ page.frontmatter.author }}</div>
        </div>
      </li>
    </router-link>
  </ol>
</template>
<script>
export default {
  data() {
    return {
      pages: []
    }
  },
  mounted() {
    this.$site.pages.forEach(page => {
      //console.log(page);
      //console.log(page.frontmatter.type);
      if (page.frontmatter.type === 'post') {
        this.pages.push(page)
      }
    })
  }
}
</script>
