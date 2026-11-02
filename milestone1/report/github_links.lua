-- Rewrite relative links to absolute GitHub URLs so they work inside the PDF.
local base = "https://github.com/anshuman-dataverse/text-to-image-generation/blob/main/milestone1/report/"
local function normalize(path)
  local parts = {}
  for seg in (base .. path):gsub("^https://", ""):gmatch("[^/]+") do
    if seg == ".." then table.remove(parts) elseif seg ~= "." then table.insert(parts, seg) end
  end
  return "https://" .. table.concat(parts, "/")
end
function Link(el)
  if not el.target:match("^%a[%w+.-]*:") and not el.target:match("^#") then
    el.target = normalize(el.target)
  end
  return el
end
