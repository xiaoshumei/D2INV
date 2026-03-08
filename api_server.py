from flask import (
    Flask,
    request,
    jsonify,
    Response,
    send_from_directory,
    abort,
    render_template_string,
    redirect,
    url_for,
)
from flask_cors import CORS
import json
from api.app import d2inv
import os
import re

app = Flask(__name__)
CORS(app)
app.static_folder = "web"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@app.route("/api/d2inv_stream", methods=["GET"])
def d2inv_stream():
    try:
        dataset_name = request.args.get("dataset_name")
        if dataset_name:
            unique_filename = dataset_name
        else:
            return jsonify({"error": "No dataset provided"}), 400

        def generate():
            for result in d2inv(unique_filename):
                yield f"data: {json.dumps(result)}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/list_datasets", methods=["GET"])
def list_datasets():
    """get all available datasets"""
    try:
        import os

        datasets_dir = "./datasets"
        if os.path.exists(datasets_dir):
            files = os.listdir(datasets_dir)
            return jsonify(files)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.before_request
def normalize_path():
    # 只处理/gallery开头的请求
    if request.path.startswith("/gallery"):
        # 将多个连续斜杠替换为单个斜杠
        normalized_path = re.sub(r"/+", "/", request.path)

        # 如果路径被改变了，重定向到规范化后的URL
        if normalized_path != request.path:
            return redirect(normalized_path, 301)
    return None


@app.route("/gallery")
def gallery_root():
    """显示results根目录的索引页面"""
    try:
        # 检查目录是否存在
        if not os.path.exists(RESULTS_DIR):
            abort(404, description="Results directory not found")

        # 获取results目录中的所有文件和子目录
        items = os.listdir(RESULTS_DIR)

        # 分离文件和目录
        directories = []
        files = []

        for item in items:
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isdir(item_path):
                directories.append(item)
            else:
                files.append(item)

        # 生成HTML索引页面
        return render_template_string(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results Gallery - Root</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }
                h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 8px 0; padding: 5px 10px; border-radius: 4px; }
                li:hover { background-color: #f5f5f5; }
                a { text-decoration: none; color: #0366d6; }
                a:hover { text-decoration: underline; }
                .directory { font-weight: 500; }
                .directory::before { content: "📁 "; margin-right: 5px; }
                .file::before { content: "📄 "; margin-right: 5px; }
                .image::before { content: "🖼️ "; }
                .pdf::before { content: "📕 "; }
                .html::before { content: "🌐 "; }
                .stats { color: #666; font-size: 0.9em; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; }
            </style>
        </head>
        <body>
            <h1>Results Gallery - Root Directory</h1>
            
            {% if directories %}
            <h2>Directories</h2>
            <ul>
                {% for dir in directories|sort %}
                <li class="directory">
                    <a href="{{ url_for('gallery_item', item_path=dir) }}/">{{ dir }}/</a>
                </li>
                {% endfor %}
            </ul>
            {% endif %}
            
            {% if files %}
            <h2>Files</h2>
            <ul>
                {% for file in files|sort %}
                <li class="file 
                    {%- if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')) %}image
                    {%- elif file.lower().endswith('.pdf') %}pdf
                    {%- elif file.lower().endswith(('.html', '.htm')) %}html
                    {%- endif -%}">
                    <a href="{{ url_for('gallery_item', item_path=file) }}">{{ file }}</a>
                </li>
                {% endfor %}
            </ul>
            {% endif %}
            
            {% if not directories and not files %}
            <p>Empty directory</p>
            {% endif %}
            
            <div class="stats">
                Total: {{ directories|length }} directories, {{ files|length }} files
            </div>
        </body>
        </html>
        """,
            directories=directories,
            files=files,
        )

    except Exception as e:
        print(f"Error in gallery_root: {str(e)}")
        abort(500)


@app.route("/gallery/<path:item_path>")
def gallery_item(item_path):
    """处理所有gallery路径，根据类型返回文件或目录"""
    try:
        # 移除路径中可能的多余斜杠
        item_path = item_path.strip("/")

        # 如果路径为空，重定向到根目录
        if not item_path:
            return redirect(url_for("gallery_root"))

        # 构建完整路径
        full_path = os.path.join(RESULTS_DIR, item_path)

        # 检查路径是否存在
        if not os.path.exists(full_path):
            abort(404)

        # 如果是目录
        if os.path.isdir(full_path):
            # 检查URL是否以斜杠结尾
            if not request.path.endswith("/"):
                # 如果没有斜杠，添加斜杠并重定向
                return redirect(request.path + "/", 301)

            # 显示目录内容
            items = os.listdir(full_path)

            # 分离文件和目录
            directories = []
            files = []

            for item in items:
                item_full_path = os.path.join(full_path, item)
                if os.path.isdir(item_full_path):
                    directories.append(item)
                else:
                    files.append(item)

            # 生成面包屑导航
            parts = item_path.split("/")
            breadcrumbs = []
            current_path = ""
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    current_path += part + "/"
                    breadcrumbs.append(
                        (
                            part,
                            url_for("gallery_item", item_path=current_path.rstrip("/")),
                        )
                    )
                else:
                    breadcrumbs.append((part, None))

            # 生成HTML索引页面
            return render_template_string(
                """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Results Gallery - {{ item_path }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }
                    h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                    .breadcrumb { padding: 10px 0; margin-bottom: 20px; background-color: #f8f9fa; border-radius: 4px; padding-left: 10px; }
                    .breadcrumb a { color: #0366d6; text-decoration: none; }
                    .breadcrumb a:hover { text-decoration: underline; }
                    .breadcrumb span { color: #666; }
                    ul { list-style-type: none; padding: 0; }
                    li { margin: 8px 0; padding: 5px 10px; border-radius: 4px; }
                    li:hover { background-color: #f5f5f5; }
                    a { text-decoration: none; color: #0366d6; }
                    a:hover { text-decoration: underline; }
                    .directory { font-weight: 500; }
                    .directory::before { content: "📁 "; margin-right: 5px; }
                    .file::before { content: "📄 "; margin-right: 5px; }
                    .image::before { content: "🖼️ "; }
                    .pdf::before { content: "📕 "; }
                    .html::before { content: "🌐 "; }
                    .parent { margin-bottom: 15px; }
                    .parent::before { content: "⬆️ "; }
                    .stats { color: #666; font-size: 0.9em; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; }
                </style>
            </head>
            <body>
                <div class="breadcrumb">
                    <a href="{{ url_for('gallery_root') }}">Root</a>
                    {% if breadcrumbs %}
                        {% for name, url in breadcrumbs %}
                            » 
                            {% if url %}
                                <a href="{{ url }}/">{{ name }}</a>
                            {% else %}
                                <span>{{ name }}</span>
                            {% endif %}
                        {% endfor %}
                    {% endif %}
                </div>
                
                <h1>Directory: {{ item_path }}/</h1>
                
                <ul>
                    <li class="parent"><a href="
                        {%- if breadcrumbs -%}
                            {% if breadcrumbs|length > 1 %}
                                {{ url_for('gallery_item', item_path='/'.join(parts[:-1])) }}/
                            {% else %}
                                {{ url_for('gallery_root') }}
                            {% endif %}
                        {%- else -%}
                            {{ url_for('gallery_root') }}
                        {%- endif -%}
                    ">.. (Parent directory)</a></li>
                </ul>
                
                {% if directories %}
                <h2>Directories</h2>
                <ul>
                    {% for dir in directories|sort %}
                    <li class="directory">
                        <a href="{{ url_for('gallery_item', item_path=item_path + '/' + dir) }}/">{{ dir }}/</a>
                    </li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if files %}
                <h2>Files</h2>
                <ul>
                    {% for file in files|sort %}
                    <li class="file 
                        {%- if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')) %}image
                        {%- elif file.lower().endswith('.pdf') %}pdf
                        {%- elif file.lower().endswith(('.html', '.htm')) %}html
                        {%- endif -%}">
                        <a href="{{ url_for('gallery_item', item_path=item_path + '/' + file) }}">{{ file }}</a>
                    </li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                <div class="stats">
                    Total: {{ directories|length }} directories, {{ files|length }} files
                </div>
            </body>
            </html>
            """,
                item_path=item_path,
                directories=directories,
                files=files,
                breadcrumbs=breadcrumbs,
                parts=parts,
            )

        # 如果是文件
        else:
            # 检查URL是否以斜杠结尾（文件URL不应该以斜杠结尾）
            if request.path.endswith("/"):
                # 移除斜杠并重定向
                return redirect(request.path.rstrip("/"), 301)

            # 发送文件
            return send_from_directory(
                RESULTS_DIR, item_path, as_attachment=False, conditional=True
            )

    except FileNotFoundError:
        abort(404)
    except Exception as e:
        print(f"Error in gallery_item: {str(e)}")
        import traceback

        traceback.print_exc()
        abort(500)


# 添加错误处理器
@app.errorhandler(404)
def not_found_error(error):
    return (
        render_template_string(
            """
    <!DOCTYPE html>
    <html>
    <head>
        <title>404 - Not Found</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            h1 { color: #e74c3c; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>404 - File Not Found</h1>
        <p>The requested file or directory could not be found.</p>
        <p><a href="{{ url_for('gallery_root') }}">← Back to Gallery Root</a></p>
    </body>
    </html>
    """
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        render_template_string(
            """
    <!DOCTYPE html>
    <html>
    <head>
        <title>500 - Internal Server Error</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            h1 { color: #e74c3c; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>500 - Internal Server Error</h1>
        <p>Something went wrong on our end.</p>
        <p><a href="{{ url_for('gallery_root') }}">← Back to Gallery Root</a></p>
    </body>
    </html>
    """
        ),
        500,
    )


@app.route("/<path:filename>")
def serve_static(filename):
    file_path = os.path.join("web", filename)
    if os.path.exists(file_path):
        return send_from_directory("web", filename)
    else:
        return send_from_directory("web", "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
