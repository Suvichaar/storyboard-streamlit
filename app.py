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

# Load environment variables (.env if present)
load_dotenv()

# ===== Azure OpenAI (from secrets) =====
AOAI_ENDPOINT     = st.secrets["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY      = st.secrets["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION  = st.secrets.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
GPT_DEPLOYMENT    = st.secrets.get("GPT_DEPLOYMENT", "gpt-5-chat")

# Azure OpenAI client
client = AzureOpenAI(
    api_key=AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

# ----------- AWS S3 config -------------
aws_access_key = st.secrets["AWS_ACCESS_KEY"]
aws_secret_key = st.secrets["AWS_SECRET_KEY"]
region_name    = st.secrets["AWS_REGION"]
bucket_name    = st.secrets["AWS_BUCKET"]
s3_prefix      = st.secrets["S3_PREFIX"]
cdn_base_url   = st.secrets["CDN_BASE"]

# ensure prefix ends with a single slash
if s3_prefix and not s3_prefix.endswith("/"):
    s3_prefix = s3_prefix + "/"

# This is used by your image-resize CDN endpoint
cdn_prefix_media = "https://media.suvichaar.org/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name,
)

# ---------- Helpers ----------
def generate_slug_and_urls(title: str):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    slug = ''.join(
        c for c in title.lower()
        .replace(" ", "-").replace("_", "-")
        if c in string.ascii_lowercase + string.digits + '-'
    ).strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}"  # urlslug -> slug_nano.html
    return (
        nano,
        slug_nano,
        f"https://suvichaar.org/stories/{slug_nano}",
        f"https://stories.suvichaar.org/{slug_nano}.html",
    )

def extract_json_block(text: str) -> str:
    """
    Extract first valid JSON object from text (handles markdown fenced blocks too).
    """
    if not text:
        raise ValueError("Empty generation output")
    # Try fenced ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try any {...} top-level object
    m2 = re.search(r"(\{(?:[^{}]|(?1))*\})", text, flags=re.DOTALL)  # recursive-ish via backref
    if m2:
        return m2.group(1).strip()
    # Last resort: assume whole text is JSON
    return text.strip()

def generate_metadata(title: str) -> dict:
    """
    Ask Azure OpenAI for strict JSON only:
    {
      "meta_description": "...",
      "meta_keywords": ["...", "..."],
      "filter_tags": ["...", "..."]
    }
    """
    system = {
        "role": "system",
        "content": (
            "You are an assistant that ONLY returns strict JSON. "
            "Do not include explanations or markdown fences unless asked."
        )
    }
    user = {
        "role": "user",
        "content": (
            "For a web story, produce concise SEO metadata as strict JSON with keys:\n"
            "meta_description (<= 160 chars), meta_keywords (array of 5-12 concise lowercase phrases), "
            "filter_tags (array of 5-12 short tags). "
            f"Title: {title}\n"
            "Example format:\n"
            "{\n"
            "  \"meta_description\": \"...\",\n"
            "  \"meta_keywords\": [\"...\", \"...\"],\n"
            "  \"filter_tags\": [\"...\", \"...\"]\n"
            "}\n"
            "Return ONLY JSON."
        )
    }
    resp = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[system, user],
        max_tokens=300,
        temperature=0.5,
    )
    raw = resp.choices[0].message.content
    json_str = extract_json_block(raw)
    data = json.loads(json_str)
    # Normalize
    desc = str(data.get("meta_description", "")).strip()
    kws = data.get("meta_keywords", [])
    tags = data.get("filter_tags", [])
    if isinstance(kws, str):
        kws_list = [x.strip() for x in kws.split(",") if x.strip()]
    else:
        kws_list = [str(x).strip() for x in kws if str(x).strip()]
    if isinstance(tags, str):
        tag_list = [x.strip() for x in tags.split(",") if x.strip()]
    else:
        tag_list = [str(x).strip() for x in tags if str(x).strip()]
    return {
        "meta_description": desc[:160],  # enforce soft limit
        "meta_keywords_csv": ", ".join(kws_list),
        "filter_tags_csv": ", ".join(tag_list),
    }

# ---------- UI ----------
st.title("Content Submission Form")

# Session defaults
if "last_title" not in st.session_state:
    st.session_state.last_title = ""
if "meta_description" not in st.session_state:
    st.session_state.meta_description = ""
if "meta_keywords" not in st.session_state:
    st.session_state.meta_keywords = ""
if "generated_filter_tags" not in st.session_state:
    st.session_state.generated_filter_tags = ""

# Title input
story_title = st.text_input("Story Title")

# Auto-generate metadata when title changes
if story_title.strip() and story_title != st.session_state.last_title:
    with st.spinner("Generating meta description, keywords, and filter tags..."):
        try:
            md = generate_metadata(story_title.strip())
            st.session_state.meta_description = md["meta_description"]
            st.session_state.meta_keywords = md["meta_keywords_csv"]
            st.session_state.generated_filter_tags = md["filter_tags_csv"]
        except Exception as e:
            st.warning(f"Metadata generation failed: {e}")
    st.session_state.last_title = story_title

with st.form("content_form"):
    # Editable, prefilled fields
    meta_description = st.text_area(
        "Meta Description",
        value=st.session_state.meta_description,
        help="Aim for ~150–160 characters."
    )
    meta_keywords = st.text_input(
        "Meta Keywords (comma separated)",
        value=st.session_state.meta_keywords
    )
    content_type = st.selectbox("Select your contenttype", ["News", "Article"])
    language = st.selectbox("Select your Language", ["en-US", "hi"])
    image_url = st.text_input("Enter your Image URL")
    html_file = st.file_uploader("Upload your Raw HTML File", type=["html", "htm"])
    categories = st.selectbox(
        "Select your Categories",
        ["Art", "Travel", "Entertainment", "Literature", "Books", "Sports", "History", "Culture", "Wildlife", "Spiritual", "Food"]
    )

    # Tags (prefilled from generator, still editable)
    default_tags = [
        "Lata Mangeshkar", "Indian Music Legends", "Playback Singing", "Bollywood Golden Era",
        "Indian Cinema", "Musical Icons", "Voice of India", "Bharat Ratna",
        "Indian Classical Music", "Hindi Film Songs", "Legendary Singers",
        "Cultural Heritage", "Suvichaar Stories"
    ]
    tag_input = st.text_input(
        "Filter Tags (comma separated)",
        value=st.session_state.get("generated_filter_tags", ", ".join(default_tags)),
        help="Example: music, culture, lata mangeshkar"
    )

    use_custom_cover = st.radio("Do you want to add a custom cover image URL?", ("No", "Yes"))
    if use_custom_cover == "Yes":
        cover_image_url = st.text_input("Enter your custom Cover Image URL")
    else:
        cover_image_url = image_url  # fallback to image_url

    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Validation before processing
    missing_fields = []
    if not story_title.strip():      missing_fields.append("Story Title")
    if not meta_description.strip(): missing_fields.append("Meta Description")
    if not meta_keywords.strip():    missing_fields.append("Meta Keywords")
    if not content_type.strip():     missing_fields.append("Content Type")
    if not language.strip():         missing_fields.append("Language")
    if not image_url.strip():        missing_fields.append("Image URL")
    if not tag_input.strip():        missing_fields.append("Filter Tags")
    if not categories.strip():       missing_fields.append("Category")
    if not html_file:                missing_fields.append("Raw HTML File")

    if missing_fields:
        st.error("❌ Please fill all required fields before submitting:\n- " + "\n- ".join(missing_fields))
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

        filter_tags_list = [tag.strip() for tag in tag_input.split(",") if tag.strip()]

        category_mapping = {
            "Art": 1, "Travel": 2, "Entertainment": 3, "Literature": 4, "Books": 5,
            "Sports": 6, "History": 7, "Culture": 8, "Wildlife": 9, "Spiritual": 10, "Food": 11
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

        # If img already on media.suvichaar.org, generate resized variants via CDN function
        if image_url.startswith("http://media.suvichaar.org") or image_url.startswith("https://media.suvichaar.org"):
            html_template = html_template.replace("{{image0}}", image_url)

            parsed_cdn_url = urlparse(image_url)
            cdn_key_path = parsed_cdn_url.path.lstrip("/")  # e.g., media/.../image.jpg

            resize_presets = {
                "potraitcoverurl": (640, 853),
                "msthumbnailcoverurl": (300, 300),
            }

            for label, (width, height) in resize_presets.items():
                template = {
                    "bucket": bucket_name,
                    "key": cdn_key_path,
                    "edits": {"resize": {"width": width, "height": height, "fit": "cover"}}
                }
                encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                final_url = f"{cdn_prefix_media}{encoded}"
                html_template = html_template.replace(f"{{{label}}}", final_url)

        # Cleanup step to remove incorrect {url} wrapping
        html_template = re.sub(r'href="\{(https://[^}]+)\}"', r'href="\1"', html_template)
        html_template = re.sub(r'src="\{(https://[^}]+)\}"', r'src="\1"', html_template)

        # ----------- Extract <style amp-custom> + <amp-story-page> from uploaded raw HTML -------------
        extracted_style = ""
        extracted_amp_story = ""
        if html_file:
            raw_html = html_file.read().decode("utf-8")

            style_match = re.search(r"(<style\s+amp-custom[^>]*>.*?</style>)", raw_html, re.DOTALL | re.IGNORECASE)
            if style_match:
                extracted_style = style_match.group(1)
            else:
                st.info("No <style amp-custom> block found in uploaded HTML.")

            start = raw_html.find("<amp-story-page")
            end = raw_html.rfind("</amp-story-page>")
            if start != -1 and end != -1:
                extracted_amp_story = raw_html[start:end + len("</amp-story-page>")]
            else:
                st.warning("No complete <amp-story> block found in uploaded HTML.")

        if extracted_style:
            head_close_pos = html_template.lower().find("</head>")
            if head_close_pos != -1:
                html_template = (
                    html_template[:head_close_pos] + "\n" + extracted_style + "\n" + html_template[head_close_pos:]
                )
            else:
                st.warning("No </head> tag found in HTML template to insert <style amp-custom>.")

        if extracted_amp_story:
            amp_story_opening_match = re.search(r"<amp-story\b[^>]*>", html_template)
            analytics_tag = '<amp-story-auto-analytics gtag-id="G-2D5GXVRK1E" class="i-amphtml-layout-container" i-amphtml-layout="container"></amp-story-auto-analytics>'
            if amp_story_opening_match and analytics_tag in html_template:
                insert_pos = amp_story_opening_match.end()
                html_template = (
                    html_template[:insert_pos]
                    + "\n\n"
                    + extracted_amp_story
                    + "\n\n"
                    + html_template[insert_pos:]
                )
            else:
                st.warning("Could not find insertion points in the HTML template.")

        # ----------- Generate and Provide Metadata JSON -------------
        metadata_dict = {
            "story_title": story_title,
            "categories": filternumber,
            "filterTags": filter_tags_list,
            "story_uid": nano,
            "story_link": canurl,
            "storyhtmlurl": canurl1,
            "urlslug": slug_nano,
            "cover_image_link": cover_image_url,
            "publisher_id": 1,
            "story_logo_link": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
            "keywords": meta_keywords,
            "metadescription": meta_description,
            "lang": language
        }

        # Upload HTML to S3 (stories bucket is distinct from media bucket)
        s3_key = f"{slug_nano}.html"
        s3_client.put_object(
            Bucket="suvichaarstories",  # <-- replace with bucket_name if you want same bucket as media
            Key=s3_key,
            Body=html_template.encode("utf-8"),
            ContentType="text/html",
        )

        final_story_url = f"https://suvichaar.org/stories/{slug_nano}"  # same as canurl
        st.success("✅ HTML uploaded successfully to S3!")
        st.markdown(f"🔗 **Live Story URL:** [Click to view your story]({final_story_url})")

        # Build ZIP with HTML + metadata for download
        json_str = json.dumps(metadata_dict, indent=4)
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
