# A minimal JSON writer (no package dependency) for the probe records.
function jsonchar(c::Char)
    c == '"' && return "\\\""
    c == '\\' && return "\\\\"
    c == '\n' && return "\\n"
    c == '\t' && return "\\t"
    c < ' ' && return "\\u" * lpad(string(UInt32(c); base = 16), 4, '0')
    return string(c)
end
json(x::AbstractString) = "\"" * join(jsonchar(c) for c in x) * "\""
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
json(::Nothing) = "null"
json(x::AbstractVector) = "[" * join(map(json, x), ", ") * "]"
json(d::AbstractDict) = "{" * join(["$(json(string(k))): $(json(v))" for (k, v) in d], ",\n ") * "}"
json(x) = json(string(x))

write_json(path, d) = (open(io -> println(io, json(d)), path, "w"); path)
