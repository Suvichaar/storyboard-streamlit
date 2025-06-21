import os
import uuid
import random
import json
import base64
import string
import streamlit as st
import boto3
import requests
from urllib.parse import urlparse
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone
import re
from io import BytesIO
import zipfile
# Load environment variables
load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-01-01-preview",
)

# ----------- AWS S3 config -------------
aws_access_key = st.secrets["AWS_ACCESS_KEY"]
aws_secret_key = st.secrets["AWS_SECRET_KEY"]
region_name = st.secrets["AWS_REGION"]
bucket_name = st.secrets["AWS_BUCKET"]
s3_prefix = st.secrets["S3_PREFIX"]
cdn_base_url = st.secrets["CDN_BASE"]
cdn_prefix_media = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name,
)

# Slug and URL generator
def generate_slug_and_urls(title):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    slug = ''.join(c for c in title.lower().replace(" ", "-").replace("_", "-") if c in string.ascii_lowercase + string.digits + '-')
    slug = slug.strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}" # this is the urlslug -> slug_nano.html
    return nano, slug_nano, f"https://suvichaar.org/stories/{slug_nano}", f"https://stories.suvichaar.org/{slug_nano}.html"

# Sidebar Chat
with st.sidebar:
    st.header("Azure OpenAI Chat")
    user_question = st.text_input("Your question:")
    if st.button("Send"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Waiting for response..."):
                messages = [{"role": "user", "content": user_question}]
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.5,
                )
                st.success("Answer:")
                st.write(response.choices[0].message.content)

# Content Submission Form
st.title("Content Submission Form")
if "last_title" not in st.session_state:
    st.session_state.last_title = ""
    st.session_state.meta_description = ""
    st.session_state.meta_keywords = ""

# Title input outside form for dynamic update
story_title = st.text_input("Story Title")

# Auto-generate metadata if story_title changed

if story_title.strip() and story_title != st.session_state.last_title:
    with st.spinner("Generating meta description, keywords, and filter tags..."):
        messages = [
            {
                "role": "user",
                "content": f"""
                Generate the following for a web story titled '{story_title}':
                1. A short SEO-friendly meta description
                2. Meta keywords (comma separated)
                3. Relevant filter tags (comma separated, suitable for categorization and content filtering)"""
            }
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=300,
                temperature=0.5,
            )
            output = response.choices[0].message.content

            # Extract metadata using regex
            desc = re.search(r"[Dd]escription\s*[:\-]\s*(.+)", output)
            keys = re.search(r"[Kk]eywords\s*[:\-]\s*(.+)", output)
            tags = re.search(r"[Ff]ilter\s*[Tt]ags\s*[:\-]\s*(.+)", output)

            st.session_state.meta_description = desc.group(1).strip() if desc else ""
            st.session_state.meta_keywords = keys.group(1).strip() if keys else ""
            st.session_state.generated_filter_tags = tags.group(1).strip() if tags else ""

        except Exception as e:
            st.warning(f"Error: {e}")
        st.session_state.last_title = story_title


with st.form("content_form"):
    meta_description = st.text_area("Meta Description", value=st.session_state.meta_description)
    meta_keywords = st.text_input("Meta Keywords (comma separated)", value=st.session_state.meta_keywords)
    content_type = st.selectbox("Select your contenttype", ["News", "Article"])
    language = st.selectbox("Select your Language", ["en-US", "hi"])
    image_url = st.text_input("Enter your Image URL")
    html_file = st.file_uploader("Upload your Raw HTML File", type=["html", "htm"])
    categories = st.selectbox("Select your Categories", ["Art", "Travel", "Entertainment", "Literature", "Books", "Sports", "History", "Culture", "Wildlife", "Spiritual", "Food"])
    # Input field
    default_tags = [
        "Lata Mangeshkar",
        "Indian Music Legends",
        "Playback Singing",
        "Bollywood Golden Era",
        "Indian Cinema",
        "Musical Icons",
        "Voice of India",
        "Bharat Ratna",
        "Indian Classical Music",
        "Hindi Film Songs",
        "Legendary Singers",
        "Cultural Heritage",
        "Suvichaar Stories"
    ]

    tag_input = st.text_input(
        "Enter Filter Tags (comma separated):",
        value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)),
        help="Example: Music, Culture, Lata Mangeshkar"
    )


    use_custom_cover = st.radio("Do you want to add a custom cover image URL?", ("No", "Yes"))
    if use_custom_cover == "Yes":
        cover_image_url = st.text_input("Enter your custom Cover Image URL")
    else:
        cover_image_url = image_url  # fallback to image_url
    # Select a user randomly and map to profile URL
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Validation before processing
    missing_fields = []

    if not story_title.strip():
        missing_fields.append("Story Title")
    if not meta_description.strip():
        missing_fields.append("Meta Description")
    if not meta_keywords.strip():
        missing_fields.append("Meta Keywords")
    if not content_type.strip():
        missing_fields.append("Content Type")
    if not language.strip():
        missing_fields.append("Language")
    if not image_url.strip():
        missing_fields.append("Image URL")
    if not tag_input.strip():
        missing_fields.append("Filter Tags")
    if not categories.strip():
        missing_fields.append("Category")
    if not html_file:
        missing_fields.append("Raw HTML File")

    if missing_fields:
        st.error(f"❌ Please fill all required fields before submitting:\n- " + "\n- ".join(missing_fields))
    else:
        # ✅ All fields are valid, proceed with your full processing logic
        st.markdown("### Submitted Data")
        st.write(f"**Story Title:** {story_title}")
        st.write(f"**Meta Description:** {meta_description}")
        st.write(f"**Meta Keywords:** {meta_keywords}")
        st.write(f"**Content Type:** {content_type}")
        st.write(f"**Language:** {language}")

    key_path = "media/default.png"
    uploaded_url = ""

    try:
        nano, slug_nano, canurl, canurl1 = generate_slug_and_urls(story_title)
        page_title = f"{story_title} | Suvichaar"
    except Exception as e:

        st.error(f"Error generating canonical URLs: {e}")
        nano = slug_nano = canurl = canurl1 = page_title = ""

    # Image URL handling
    if image_url:

        filename = os.path.basename(urlparse(image_url).path)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".gif"]:
            ext = ".jpg"

        if image_url.startswith("https://stories.suvichaar.org/"):

            uploaded_url = image_url
            key_path = "/".join(urlparse(image_url).path.split("/")[2:])

        else:

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                unique_filename = f"{uuid.uuid4().hex}{ext}"
                s3_key = f"{s3_prefix}{unique_filename}"
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=response.content,
                    ContentType=response.headers.get("Content-Type", "image/jpeg"),
                )
                uploaded_url = f"{cdn_base_url}{s3_key}"
                key_path = s3_key
                st.success("Image uploaded successfully!")

            except Exception as e:
                st.warning(f"Failed to fetch/upload image. Using fallback. Error: {e}")
                uploaded_url = ""
    else:
        st.info("No Image URL provided. Using default.")

    try:
        template_path = "templates/masterregex.html"
        with open(template_path, "r", encoding="utf-8") as file:
            html_template = file.read()


        user_mapping = {
            "Mayank": "https://www.instagram.com/iamkrmayank?igsh=eW82NW1qbjh4OXY2&utm_source=qr",
            "Onip": "https://www.instagram.com/onip.mathur/profilecard/?igsh=MW5zMm5qMXhybGNmdA==",
            "Naman": "https://njnaman.in/"
        }

        filter_tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]

        category_mapping = {
            "Art": 21,
            "Travel": 22,
            "Entertainment": 23,
            "Literature": 24,
            "Books": 25,
            "Sports": 26,
            "History": 27,
            "Culture": 28,
            "Wildlife": 29,
            "Spiritual": 30
        }

        filternumber = category_mapping[categories]
        selected_user = random.choice(list(user_mapping.keys()))
        html_template = html_template.replace("{{user}}", selected_user)
        html_template = html_template.replace("{{userprofileurl}}", user_mapping[selected_user])
        html_template = html_template.replace("{{publishedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
        html_template = html_template.replace("{{modifiedtime}}", datetime.now(timezone.utc).isoformat(timespec='seconds'))
        html_template = html_template.replace("{{storytitle}}", story_title)
        html_template = html_template.replace("{{metadescription}}", meta_description)
        html_template = html_template.replace("{{metakeywords}}", meta_keywords)
        html_template = html_template.replace("{{contenttype}}", content_type)
        html_template = html_template.replace("{{lang}}", language)
        html_template = html_template.replace("{{pagetitle}}", page_title)
        html_template = html_template.replace("{{canurl}}", canurl)
        html_template = html_template.replace("{{canurl1}}", canurl1)

        if image_url.startswith("http://media.suvichaar.org") or image_url.startswith("https://media.suvichaar.org"):
        
            html_template = html_template.replace("{{image0}}", image_url)

            parsed_cdn_url = urlparse(image_url)
            cdn_key_path = parsed_cdn_url.path.lstrip("/")  # ✅ Fix

            resize_presets = {
                "potraitcoverurl": (640, 853),
                "msthumbnailcoverurl": (300, 300),
            }

            for label, (width, height) in resize_presets.items():
                template = {
                    "bucket": bucket_name,
                    "key": cdn_key_path,
                    "edits": {
                        "resize": {
                            "width": width,
                            "height": height,
                            "fit": "cover"
                        }
                    }
                }
                encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                final_url = f"{cdn_prefix_media}{encoded}"
                # st.write(f"✅ Replacing {{{label}}} with {final_url}")
                html_template = html_template.replace(f"{{{label}}}", final_url)

        # Cleanup step to remove incorrect {url} wrapping
        html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\1"', html_template)
        html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\1"', html_template)

        # ----------- Extract <style amp-custom> block from uploaded raw HTML -------------
        extracted_style = ""
        if html_file:
            raw_html = html_file.read().decode("utf-8")

            # Extract <style amp-custom> block
            style_match = re.search(r"(<style\s+amp-custom[^>]*>.*?</style>)", raw_html, re.DOTALL | re.IGNORECASE)
            if style_match:
                extracted_style = style_match.group(1)
            else:
                st.info("No <style amp-custom> block found in uploaded HTML.")

            # Extract <amp-story> block
            start = raw_html.find("<amp-story-page")
            end = raw_html.rfind("</amp-story-page>")
            extracted_amp_story = ""
            if start != -1 and end != -1:
                extracted_amp_story = raw_html[start:end + len("</amp-story-page>")]
            else:
                st.warning("No complete <amp-story> block found in uploaded HTML.")
        else:
            extracted_amp_story = ""

        # Insert extracted <style amp-custom> into <head> of your template before </head>
        if extracted_style:
            head_close_pos = html_template.lower().find("</head>")
            if head_close_pos != -1:
                html_template = (
                    html_template[:head_close_pos] +
                    "\n" + extracted_style + "\n" +
                    html_template[head_close_pos:]
                )
            else:
                st.warning("No </head> tag found in HTML template to insert <style amp-custom>.")

        # Insert extracted AMP story block inside template
        if extracted_amp_story:
            # Locate opening <amp-story> tag in template
            amp_story_opening_match = re.search(r"<amp-story\b[^>]*>", html_template)
            analytics_tag = '<amp-story-auto-analytics gtag-id="G-2D5GXVRK1E" class="i-amphtml-layout-container" i-amphtml-layout="container"></amp-story-auto-analytics>'

            if amp_story_opening_match and analytics_tag in html_template:
                insert_pos = amp_story_opening_match.end()
                # Insert the extracted story slides just after the opening tag, before analytics tag
                html_template = (
                    html_template[:insert_pos]
                    + "\n\n"
                    + extracted_amp_story
                    + "\n\n"
                    + html_template[insert_pos:]
                )
            else:
                st.warning("Could not find insertion points in the HTML template.")

        #st.markdown("### Final Modified HTML")
        # st.code(html_template, language="html")

        # ----------- Generate and Provide Metadata JSON -------------
        metadata_dict = {
            "story_title": story_title,
            "categories": filternumber,
            "filterTags": filter_tags,
            "story_uid": nano,
            "story_link": canurl,
            "storyhtmlurl": canurl1,
            "urlslug": slug_nano,
            "cover_image_link": cover_image_url,
            "publisher_id": 3,
            "story_logo_link": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
            "keywords": meta_keywords,
            "metadescription": meta_description,
            "lang": language
        }

        s3_key = f"{slug_nano}.html"

        s3_client.put_object(
            Bucket="suvichaarstories",
            Key=s3_key,
            Body=html_template.encode("utf-8"),
            ContentType="text/html",
        )

        final_story_url = f"https://suvichaar.org/stories/{slug_nano}"  # This is your canurl
        st.success("✅ HTML uploaded successfully to S3!")
        st.markdown(f"🔗 **Live Story URL:** [Click to view your story]({final_story_url})")
        
        json_str = json.dumps(metadata_dict, indent=4)

        # Save data to session_state
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr(f"{slug_nano}.html", html_template)
            zip_file.writestr(f"{slug_nano}_metadata.json", json_str)
        
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download HTML + Metadata ZIP",
            data=zip_buffer,
            file_name=f"{story_title}.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Error processing HTML: {e}")
