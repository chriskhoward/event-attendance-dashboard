<?php
/**
 * Template Name: Event Analysis Dashboard
 * 
 * This is a custom template for the Event Analysis Dashboard.
 */

get_header(); ?>

<div class="event-analysis-container">
    <div class="event-analysis-header">
        <h1><?php echo get_the_title(); ?></h1>
        <?php if (have_posts()) : while (have_posts()) : the_post(); ?>
            <div class="event-analysis-description">
                <?php the_content(); ?>
            </div>
        <?php endwhile; endif; ?>
    </div>

    <div class="event-analysis-iframe-container">
        <iframe
            src="https://your-app-name.streamlit.app"
            width="100%"
            height="800px"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="border: none;"
        ></iframe>
    </div>
</div>

<style>
.event-analysis-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.event-analysis-header {
    text-align: center;
    margin-bottom: 30px;
}

.event-analysis-description {
    margin: 20px 0;
    color: #666;
}

.event-analysis-iframe-container {
    position: relative;
    width: 100%;
    height: 800px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    overflow: hidden;
}

@media (max-width: 768px) {
    .event-analysis-iframe-container {
        height: 600px;
    }
}
</style>

<?php get_footer(); ?> 