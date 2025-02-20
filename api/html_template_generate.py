from tools.llm import get_llm
from tools.utils import preprocess_response


def generate_html_template(data_story, vendor="openai", model_type="general"):
    system_prompt = """
        ## Profile:
            - Author: D2INV
            - Version: 1.1
            - Language: English
            - Specialization: Data Visualization, Web Development
        ## Goals:
            - Generate an HTML template for data storytelling infographics.
            - Include title, subtitle, multi-section layout, chart placeholders, and adhere to infographic best practices.
            - Incorporate rich colors, icons, animations, and CTAs for engagement.
            - Ensure template is adaptable to any dataset and visually cohesive.
        ## Constraints:
            - Must use semantic HTML5 elements (header, section, article).
            - Requires AOS library for scroll-based animations.
            - Icons must be sourced from legal CDNs (FontAwesome, Icons8).
            - Placeholder IDs must follow a sequential naming convention (chart_1, chart_2).
            - Color scheme must align with the dataset's theme.
            - Cross-browser compatibility (Chrome, Firefox, Safari, Edge).
        ## Skills:
            - HTML5/CSS3 (semantic markup, responsive design).
            - JavaScript (AOS library integration).
            - Data visualization principles (hierarchy, accessibility, color theory).
            - Icon integration (SVG/FontAwesome/Icons8).
            - CSS animations and transitions.
        ## Workflow:
            1. HTML Structure:
                - Create a header with story_title and story_subtitle.
                - Each story_piece becomes a section with:
                    a) A title (H2/H3) derived from question.
                    b) Descriptive text from narration.
                    c) A placeholder div (e.g., chart_1) for visuals.
            2. Best Practices Implementation:
                - Use colored backgrounds/dividers for section separation.
                - Add explainer text under the subtitle.
                - Include a "backbone" (central line/icon) to guide flow.
                - Place CTAs between sections (styled buttons/links).
            3. Styling:
                - Define a color palette matching the dataset's theme.
                - Use flexbox/grid for vertical alignment.
                - Position icons next to titles or above sections.
            4. Interactivity:
                - Add AOS animations (fade-in, slide-up) via data-aos attributes.
                - Ensure icons load from CDNs and are accessible.
                - Style CTAs with contrasting colors and hover effects.
            5. Finalization:
                - Include sources in the footer (small text with data links).
                - Test placeholder structure, animation timing, and color contrast.
                - Validate cross-browser compatibility.
            6. Sample Code:
                <div class="section" data-aos="fade-up">  
                  <h2 class="section-title"><i class="fas fa-[icon-name]"></i> [Insight Title]</h2>  
                  <p class="narration">[Descriptive Text]</p>  
                  <div class="chart" id="chart_1"></div>  
                  <a href="#next-section" class="cta">Next Insight</a>  
                </div>  
            7. Testing Checklist:
                - Verify placeholder IDs are sequential and unique.
                - Ensure animations trigger correctly on scroll.
                - Confirm icons render properly across devices.
                - Validate color contrast ratios for accessibility.
            8. Optimization:
                - Use CSS variables for easy theming.
                - Minimize external dependencies.
                - Implement lazy loading for charts/visualizations.
    """

    messages = (
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"The data story is: \n {data_story} \n. The generated html infographic is: \n ",
            },
        ]
        if model_type == "general"
        else [
            {
                "role": "system",
                "content": f"Based on the provided data story content, design an infographic-style visualization template in HTML format. The template should include a title (corresponding to the data story's story_title), a subtitle (corresponding to the data story's story_subtitle), and multiple sections to display all story_pieces. For each story_piece, the template should display its title (corresponding to the story_piece's question), description (corresponding to the story_piece's narration), and a placeholder div element with a unique id attribute (with the value chart_index, where index is replaced by the sequence number of the story_piece, starting from 1), but without any child elements. The designed HTML visualization template must adhere to infographic best practices, such as: (1) The infographic must include 1、Explainer: Brief, non-sentence introductory text. 2、Sections: Clearly separated using lines, icons, or alternating colors. 3、Explanatory Text: Support graphics with concise descriptions. 4、Call to Action (CTA): Encourage readers to take the next step. 5、Headline: Create a compelling and concise title. 6、Subheads/Labels: Clarify sections, icons, and charts. 7、Backbone: Unifying design element (line, icon, visual) to guide readers. 8、Sources: Include brief citations at the end.  （2）The style of the infographic should match the content of the data story. (3) The infographic should use rich colors, icons (e.g., from Icons8; ensure the icon links are legal and accessible), and animation effects (e.g., the AOS library). Refer to HTML-format infographics on venngage.com and marq.com for inspiration. (4) Add Calls to Action between different sections to guide users to continue browsing. Align icons and titles vertically centered on the same line where possible.",
            },
            {
                "role": "user",
                "content": f"The content of the data story is:: \n {data_story} \n. Only return the HTML format template and Don't explain anything. The generated html infographic is:  \n ",
            },
        ]
    )
    [client, model] = get_llm(vendor=vendor, model_type=model_type)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    result = chat_completion.choices[0].message.content
    html_content = preprocess_response(result)
    return html_content


def write_html_template(dataset_name, index, result):
    with open(
        f"../evaluate_results/{dataset_name}/visualization_template_{index}.html",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(result)
