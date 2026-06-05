//! Gemma4 chat-template compatibility tests.
//!
//! These tests use a distilled fixture that mirrors the current llama.cpp
//! Gemma4 interleaved template semantics without copying the whole upstream
//! template into the repository. The goal is to pin the render-context features
//! Ocelotl must support before a Gemma4 tokenizer path can claim parity:
//! BOS token injection, `enable_thinking`, assistant→model role mapping,
//! multimodal content placeholders, and tool-call/tool-response markers.

use ocelotl_tokenizer::{ChatTemplate, ChatTemplateOptions};
use serde_json::json;

const GEMMA4_COMPAT_TEMPLATE: &str = r#"
{%- macro format_argument(argument, escape_keys=True) -%}
    {%- if argument is string -%}
        {{- '<|"|>' + argument + '<|"|>' -}}
    {%- elif argument is boolean -%}
        {{- 'true' if argument else 'false' -}}
    {%- elif argument is mapping -%}
        {{- '{' -}}
        {%- set ns = namespace(found_first=false) -%}
        {%- for key, value in argument | dictsort -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {%- if escape_keys -%}
                {{- '<|"|>' + key + '<|"|>' -}}
            {%- else -%}
                {{- key -}}
            {%- endif -%}
            :{{- format_argument(value, escape_keys=escape_keys) -}}
        {%- endfor -%}
        {{- '}' -}}
    {%- elif argument is sequence -%}
        {{- '[' -}}
        {%- for item in argument -%}
            {{- format_argument(item, escape_keys=escape_keys) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- ']' -}}
    {%- else -%}
        {{- argument -}}
    {%- endif -%}
{%- endmacro -%}
{%- set ns = namespace(prev_message_type=None) -%}
{%- set loop_messages = messages -%}
{{- bos_token -}}
{%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}
    {{- '<|turn>system\n' -}}
    {%- if enable_thinking is defined and enable_thinking -%}
        {{- '<|think|>\n' -}}
    {%- endif -%}
    {%- if messages[0]['role'] in ['system', 'developer'] -%}
        {{- messages[0]['content'] | trim -}}
        {%- set loop_messages = messages[1:] -%}
    {%- endif -%}
    {%- if tools -%}
        {%- for tool in tools -%}
            {{- '<|tool>' -}}
            {{- 'declaration:' + tool['function']['name'] -}}
            {{- '<tool|>' -}}
        {%- endfor -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {{- '<|turn>' + role + '\n' -}}
    {%- if message['tool_calls'] -%}
        {%- for tool_call in message['tool_calls'] -%}
            {%- set function = tool_call['function'] -%}
            {{- '<|tool_call>call:' + function['name'] + '{' -}}
            {%- if function['arguments'] is mapping -%}
                {%- set ns_args = namespace(found_first=false) -%}
                {%- for key, value in function['arguments'] | dictsort -%}
                    {%- if ns_args.found_first %},{% endif -%}
                    {%- set ns_args.found_first = true -%}
                    {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
                {%- endfor -%}
            {%- elif function['arguments'] is string -%}
                {{- function['arguments'] -}}
            {%- endif -%}
            {{- '}<tool_call|>' -}}
        {%- endfor -%}
        {%- set ns.prev_message_type = 'tool_call' -%}
    {%- endif -%}
    {%- if message['tool_responses'] -%}
        {%- for tool_response in message['tool_responses'] -%}
            {{- '<|tool_response>' -}}
            {{- 'response:' + (tool_response['name'] | default('unknown')) + '{' -}}
            {%- if tool_response['response'] is mapping -%}
                {%- for key, value in tool_response['response'] | dictsort -%}
                    {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
                    {%- if not loop.last %},{% endif -%}
                {%- endfor -%}
            {%- else -%}
                {{- 'value:' + format_argument(tool_response['response'], escape_keys=False) -}}
            {%- endif -%}
            {{- '}<tool_response|>' -}}
        {%- endfor -%}
        {%- set ns.prev_message_type = 'tool_response' -%}
    {%- endif -%}
    {%- if message['content'] is string -%}
        {{- message['content'] | trim -}}
    {%- elif message['content'] is sequence -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] | trim -}}
            {%- elif item['type'] == 'image' -%}
                {{- '<|image|>' -}}
            {%- elif item['type'] == 'audio' -%}
                {{- '<|audio|>' -}}
            {%- elif item['type'] == 'video' -%}
                {{- '<|video|>' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' -%}
        {{- '<|turn>model\n' -}}
    {%- endif -%}
    {%- if not enable_thinking | default(false) -%}
        {{- '<|channel>thought\n<channel|>' -}}
    {%- endif -%}
{%- endif -%}
"#;

fn render(messages: &[serde_json::Value], options: ChatTemplateOptions) -> String {
    let tmpl = ChatTemplate::from_jinja(GEMMA4_COMPAT_TEMPLATE).expect("template compiles");
    tmpl.apply_with_options(messages, &Vec::<serde_json::Value>::new(), &options)
        .expect("render succeeds")
}

#[test]
fn gemma4_chat_template_emits_bos_turns_and_default_thought_prompt() {
    let rendered = render(
        &[json!({
            "role": "user",
            "content": " describe this "
        })],
        ChatTemplateOptions {
            add_generation_prompt: true,
            enable_thinking: false,
            bos_token: Some("<bos>".to_string()),
        },
    );

    assert_eq!(
        rendered,
        "<bos><|turn>user\ndescribe this<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
}

#[test]
fn gemma4_chat_template_enable_thinking_injects_system_think_marker() {
    let rendered = render(
        &[
            json!({
                "role": "system",
                "content": " Be exact. "
            }),
            json!({
                "role": "user",
                "content": " Solve it. "
            }),
        ],
        ChatTemplateOptions {
            add_generation_prompt: true,
            enable_thinking: true,
            bos_token: Some("<bos>".to_string()),
        },
    );

    assert_eq!(
        rendered,
        "<bos><|turn>system\n<|think|>\nBe exact.<turn|>\n<|turn>user\nSolve it.<turn|>\n<|turn>model\n"
    );
}

#[test]
fn gemma4_chat_template_maps_assistant_to_model_and_media_to_placeholders() {
    let rendered = render(
        &[
            json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": " look " },
                    { "type": "image" },
                    { "type": "audio" },
                    { "type": "video" }
                ]
            }),
            json!({
                "role": "assistant",
                "content": " done "
            }),
        ],
        ChatTemplateOptions {
            add_generation_prompt: false,
            enable_thinking: false,
            bos_token: Some("<bos>".to_string()),
        },
    );

    assert_eq!(
        rendered,
        "<bos><|turn>user\nlook<|image|><|audio|><|video|><turn|>\n<|turn>model\ndone<turn|>\n"
    );
}

#[test]
fn gemma4_chat_template_accepts_tools_tool_calls_and_tool_responses() {
    let tmpl = ChatTemplate::from_jinja(GEMMA4_COMPAT_TEMPLATE).expect("template compiles");
    let messages = vec![
        json!({
            "role": "user",
            "content": "weather?"
        }),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "function": {
                    "name": "lookup",
                    "arguments": { "city": "Paris", "unit": "c" }
                }
            }]
        }),
        json!({
            "role": "tool",
            "content": null,
            "tool_responses": [{
                "name": "lookup",
                "response": { "ok": true, "temperature": "20" }
            }]
        }),
    ];
    let tools = vec![json!({
        "function": {
            "name": "lookup",
            "description": "Fetch weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }
        }
    })];

    let rendered = tmpl
        .apply_with_options(
            &messages,
            &tools,
            &ChatTemplateOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                bos_token: Some("<bos>".to_string()),
            },
        )
        .expect("render succeeds");

    assert!(rendered.starts_with("<bos><|turn>system\n<|tool>declaration:lookup<tool|>"));
    assert!(rendered.contains("<|turn>model\n<|tool_call>call:lookup{city:<|\"|>Paris<|\"|>,unit:<|\"|>c<|\"|>}<tool_call|><turn|>\n"));
    assert!(rendered.contains("<|turn>tool\n<|tool_response>response:lookup{ok:true,temperature:<|\"|>20<|\"|>}<tool_response|><turn|>\n"));
    assert!(
        !rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "Gemma4 template should not add a fresh model turn after a tool response"
    );
}

#[test]
fn gemma4_chat_template_supports_upstream_tool_schema_filter_surface() {
    let tmpl = ChatTemplate::from_jinja(
        "{%- for item in messages[0]['content'] | map('upper') | list -%}{{ item }};{%- endfor -%}",
    )
    .expect("template using Gemma4 upstream filter surface compiles");

    let rendered = tmpl
        .apply_with_options(
            &[json!({
                "role": "user",
                "content": ["string", "number"]
            })],
            &Vec::<serde_json::Value>::new(),
            &ChatTemplateOptions::default(),
        )
        .expect("render succeeds");

    assert_eq!(rendered, "STRING;NUMBER;");
}
