-- Rewrite relative links to absolute GitHub URLs so they work inside the PDF.
-- The document's folder (docs/<milestone>/) is passed as the `docdir` metadata field.
local repo = "https://github.com/anshuman-dataverse/text-to-image-generation/blob/main/"
local base = repo

local function normalize(path)
  local parts = {}
  for seg in (base .. path):gsub("^https://", ""):gmatch("[^/]+") do
    if seg == ".." then table.remove(parts) elseif seg ~= "." then table.insert(parts, seg) end
  end
  return "https://" .. table.concat(parts, "/")
end

function Meta(m)
  if m.docdir then base = repo .. pandoc.utils.stringify(m.docdir) end
end

function Link(el)
  if not el.target:match("^%a[%w+.-]*:") and not el.target:match("^#") then
    el.target = normalize(el.target)
  end
  return el
end

return {{Meta = Meta}, {Link = Link}}
