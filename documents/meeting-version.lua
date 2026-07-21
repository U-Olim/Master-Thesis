function Pandoc(document)
  local meeting_version =
    pandoc.utils.stringify(document.meta["meeting-version"]) == "true"
  if not meeting_version then
    return document
  end
  return document:walk({
    Div = function(div)
      if div.classes:includes("archival-only") then
        return {}
      end
    end
  })
end
